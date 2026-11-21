package dither

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"runtime"
	"strconv"
)

// copyPalette deeply copies colors and returns a new slice that is unrelated.
// Changing the passed slice will not affect the returned one in any way.
func copyPalette(p []color.Color) []color.Color {
	ret := make([]color.Color, len(p))
	for i, c := range p {
		r, g, b, a := c.RGBA()
		ret[i] = color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)}
	}
	return ret
}

// Ditherer dithers images according to the settings in the struct.
// It can be safely reused for many images, and used concurrently.
//
// Some members of the struct are public. Those members can be changed
// in-between dithering images, if you would like to dither again.
// If you change those public methods while an image is being dithered, the
// output image will have problems, so only change in-between dithering.
//
// You can only set one of Matrix, Mapper, or Special. Trying to dither when
// none or more than one of those are set will cause the function to panic.
//
// All methods can handle images with transparency, unless otherwise specified.
// Read the docs before using!
type Ditherer struct {

	// Matrix is the ErrorDiffusionMatrix for dithering.
	Matrix ErrorDiffusionMatrix

	// Mapper is the ColorMapper function for dithering.
	Mapper PixelMapper

	// Special is the special dithering algorithm that's being used. The default
	// value of 0 indicates that no special dithering algorithm is being used.
	Special SpecialDither

	// SingleThreaded controls whether the dithering happens sequentially or using
	// runtime.GOMAXPROCS(0) workers, which defaults to the number of CPUs.
	//
	// Note that error diffusion dithering (using Matrix) is sequential by nature
	// and so this field has no effect.
	//
	// Setting this to true is only useful in rare cases, like when numbers are
	// used sequentially in a PixelMapper, and the output must be deterministic.
	// Because otherwise the numbers will be retrieved in a different order each
	// time, as the goroutines call on the PixelMapper.
	SingleThreaded bool

	// Serpentine controls whether the error diffusion matrix is applied in a
	// serpentine manner, meaning that it goes right-to-left every other line.
	// This greatly reduces line-type artifacts. If a Mapper is being used this
	// field will have no effect.
	//
	// The direction is determined by the row index relative to the image
	// bounds, so an image with a non-zero origin dithers the same way as the
	// identical image at the origin.
	Serpentine bool

	// MaxPixels is the pixel budget: the maximum number of pixels an input
	// image may contain. Images with more pixels than this cause Dither,
	// DitherCopy and friends to panic before the source image is touched in
	// any way. This protects against decompression bombs and accidental
	// huge allocations.
	//
	// A zero value means DefaultMaxPixels is used. A negative value disables
	// the budget entirely.
	MaxPixels int

	// palette holds the colors the dithered image is allowed to use, in the
	// sRGB color space. It is guaranteed to only hold colors of the type
	// color.RGBA64.
	palette []color.Color

	// linearPalette holds all the palette colors, but in linear RGB space.
	linearPalette [][3]uint16
}

// DefaultMaxPixels is the default pixel budget used when Ditherer.MaxPixels
// is zero. It is 2^26 pixels (about 67 million, e.g. 8192x8192).
const DefaultMaxPixels = 1 << 26

// NewDitherer creates a new Ditherer that uses a copy of the provided palette.
// If the palette is empty or nil then nil will be returned.
// All palette colors should be opaque.
func NewDitherer(palette []color.Color) *Ditherer {
	if len(palette) == 0 {
		return nil
	}

	d := &Ditherer{}

	// Palette is copied so the user can't modify it externally later
	d.palette = copyPalette(palette)

	// Create linear RGB version of the palette
	d.linearPalette = make([][3]uint16, len(d.palette))
	for i := range d.linearPalette {
		r, g, b := toLinearRGB(d.palette[i])
		d.linearPalette[i] = [3]uint16{r, g, b}
	}

	return d
}

// invalid returns true when the current struct fields of the Ditherer make it
// impossible to dither.
func (d *Ditherer) invalid() bool {
	// This basically XORs three bools that represent whether each value is
	// unset or not. The if statement evaluates to true if one is set, but
	// false if none or more than one are set. But then it's flipped with !()
	// on the outside.
	if !((d.Mapper != nil) != ((d.Matrix != nil) != (d.Special != 0))) {
		return true
	}
	if d.Special != 0 {
		// No special dithering supported right now
		return true
	}
	return false
}

// validateImage performs pre-flight checks on the input image. It panics if
// the image cannot be safely dithered. It is always called before the source
// image is modified in any way, so a failed validation never leaves the
// source image partially dithered.
//
// The checks are:
//   - the image is not nil
//   - the number of pixels is within the pixel budget (see MaxPixels)
//   - for the standard library's concrete image types, the Pix slice and
//     Stride are consistent with the Bounds, so that reading or writing a
//     pixel cannot panic or corrupt memory halfway through
//
// Images with empty bounds are valid; dithering them is a no-op.
func (d *Ditherer) validateImage(src image.Image) {
	if src == nil {
		panic("dither: cannot dither a nil image")
	}
	b := src.Bounds()
	if b.Empty() {
		return
	}

	// Pixel budget
	limit := int64(d.MaxPixels)
	if limit == 0 {
		limit = DefaultMaxPixels
	}
	if limit > 0 {
		px := int64(b.Dx()) * int64(b.Dy())
		if px > limit {
			panic("dither: image has " + strconv.FormatInt(px, 10) + " pixels, which exceeds the pixel budget of " + strconv.FormatInt(limit, 10))
		}
	}

	if !validImageMemory(src) {
		panic("dither: image Pix slice and Stride are inconsistent with its Bounds")
	}
}

// validImageMemory checks that the standard library's concrete image types
// have Pix slices and strides that are consistent with their Bounds, so that
// every pixel inside Bounds can be read and written safely. Image types this
// package doesn't know about are assumed to be safe, since their own At/Set
// methods are responsible for bounds checking.
func validImageMemory(img image.Image) bool {
	b := img.Bounds()
	dx, dy := b.Dx(), b.Dy()

	// check verifies that pix can hold dy rows of dx pixels at bpp bytes per
	// pixel with the given stride.
	check := func(pixLen, stride, bpp int) bool {
		if stride < dx*bpp {
			return false
		}
		if dy == 0 {
			return true
		}
		return pixLen >= stride*(dy-1)+dx*bpp
	}

	switch p := img.(type) {
	case *image.Alpha:
		return check(len(p.Pix), p.Stride, 1)
	case *image.Alpha16:
		return check(len(p.Pix), p.Stride, 2)
	case *image.Gray:
		return check(len(p.Pix), p.Stride, 1)
	case *image.Gray16:
		return check(len(p.Pix), p.Stride, 2)
	case *image.CMYK:
		return check(len(p.Pix), p.Stride, 4)
	case *image.NRGBA:
		return check(len(p.Pix), p.Stride, 4)
	case *image.NRGBA64:
		return check(len(p.Pix), p.Stride, 8)
	case *image.Paletted:
		return check(len(p.Pix), p.Stride, 1)
	case *image.RGBA:
		return check(len(p.Pix), p.Stride, 4)
	case *image.RGBA64:
		return check(len(p.Pix), p.Stride, 8)
	case *image.YCbCr:
		return check(len(p.Y), p.YStride, 1) && validChroma(p)
	case *image.NYCbCrA:
		return check(len(p.Y), p.YStride, 1) && validChroma(&p.YCbCr) &&
			check(len(p.A), p.AStride, 1)
	}
	return true
}

// validChroma checks the chroma planes of a *image.YCbCr.
func validChroma(p *image.YCbCr) bool {
	b := p.Bounds()
	dx, dy := b.Dx(), b.Dy()

	var cw, ch int
	switch p.SubsampleRatio {
	case image.YCbCrSubsampleRatio444:
		cw, ch = dx, dy
	case image.YCbCrSubsampleRatio422:
		cw, ch = (dx+1)/2, dy
	case image.YCbCrSubsampleRatio420:
		cw, ch = (dx+1)/2, (dy+1)/2
	case image.YCbCrSubsampleRatio440:
		cw, ch = dx, (dy+1)/2
	case image.YCbCrSubsampleRatio411:
		cw, ch = (dx+3)/4, dy
	case image.YCbCrSubsampleRatio410:
		cw, ch = (dx+3)/4, (dy+1)/2
	default:
		return false
	}

	if len(p.Cb) != len(p.Cr) || p.CStride < cw {
		return false
	}
	if ch == 0 {
		return true
	}
	need := p.CStride*(ch-1) + cw
	return len(p.Cb) >= need && len(p.Cr) >= need
}

// GetPalette returns a copy of the current palette being used by the Ditherer.
func (d *Ditherer) GetPalette() []color.Color {
	// Palette is copied so the user can't modify it externally later
	return copyPalette(d.palette)
}

func sqDiff(v1 uint16, v2 uint16) uint32 {
	// This optimization is copied from Go stdlib, see
	// https://github.com/golang/go/blob/go1.15.7/src/image/color/color.go#L314

	d := uint32(v1) - uint32(v2)
	return (d * d) >> 2
}

// closestColor returns the index of the color in the palette that's closest to
// the provided one, using Euclidean distance in linear RGB space. The provided
// RGB values must be linear RGB.
func (d *Ditherer) closestColor(r, g, b uint16) int {
	// Go through each color and find the closest one
	color, best := 0, uint32(math.MaxUint32)
	for i, c := range d.linearPalette {

		// Euclidean distance, but the square root part is removed
		// Weight by luminance value to approximate radiant power / luminance
		// as humans perceive it.
		//
		// These values were taken from Wikipedia:
		// https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale
		// 0.2126, 0.7152, 0.0722
		// The are changed to fractions here to keep everything in integer math:
		//     1063/5000, 447/625, 361/5000
		// Unfortunately this requires promoting them to uint64 to prevent overflow

		dist := uint32(
			1063*uint64(sqDiff(r, c[0]))/5000 +
				447*uint64(sqDiff(g, c[1]))/625 +
				361*uint64(sqDiff(b, c[2]))/5000,
		)

		if dist < best {
			if dist == 0 {
				return i
			}
			color, best = i, dist
		}
	}
	return color
}

// unpremultAndLinearize unpremultiplies the provided color, and returns the
// linearized RGB values, as well as the unchanged alpha value.
func unpremultAndLinearize(c color.Color) (uint16, uint16, uint16, uint16) {
	// alpha
	var a uint16

	// Optimize for different color types
	// Opaque colors are fast-tracked
	// Non-premultiplied colors aren't unpremulted, and all others are
	switch v := c.(type) {
	case color.Gray:
		a = 0xffff
	case color.Gray16:
		a = 0xffff
	case color.NRGBA:
		// (1/255)*65535 = 257
		// This converts 8-bit color into 16-bit
		a = uint16(v.A) * 257
	case color.NRGBA64:
		a = v.A
	default:
		c = color.NRGBA64Model.Convert(c)
		_, _, _, x := c.RGBA()
		a = uint16(x)
	}

	r, g, b := toLinearRGB(c)
	return r, g, b, a
}

// premultAlpha takes the dithered color for a position and the alpha value of
// the original image at that position, and returns a color that's corrected
// to take the alpha value into account -- premultiplying it.
//
// The alpha policy, which is identical for the PixelMapper and error
// diffusion paths, is:
//   - Fully transparent pixels (alpha == 0) hold no color. They are always
//     output as transparent black (0, 0, 0, 0), and any RGB values hidden in
//     them are never treated as visible color.
//   - Partially transparent pixels are dithered based on their
//     unpremultiplied color, and the dithered color is premultiplied by the
//     original alpha on output. The alpha value itself is never dithered.
func premultAlpha(c color.RGBA64, a uint16) color.RGBA64 {
	// Algorithm described in #8
	// https://github.com/makeworld-the-better-one/dither/issues/8

	if a == 0 {
		// Transparent, no color values are held
		return color.RGBA64{0, 0, 0, 0}
	}
	if a == 0xffff {
		// Pixel is opaque, no alpha math needed
		return c
	}
	// Multiply RGB by alpha value - return premultiplied color
	// Adapted from https://github.com/golang/go/blob/go1.16.4/src/image/color/color.go#L84
	r := uint32(c.R)
	r *= uint32(a)
	r /= 0xffff
	g := uint32(c.G)
	g *= uint32(a)
	g /= 0xffff
	b := uint32(c.B)
	b *= uint32(a)
	b /= 0xffff

	return color.RGBA64{
		R: uint16(r),
		G: uint16(g),
		B: uint16(b),
		A: uint16(a),
	}
}

// Dither dithers the provided image.
//
// It will always try to change the provided image and return it, but if that
// is not possible it will return the dithered image as a copy.
//
// In comparison to DitherCopy, this can greatly reduce memory usage, and is quicker
// because it usually won't copy the image at the beginning. It should be preferred
// if you don't need to keep the original image.
//
// Cases where a copy will be are limited to:
// If the input image is *image.Paletted and the image's palette is different than
// the Ditherer's, or if the image can't be casted to draw.Image.
//
// The returned image type when copied is *image.RGBA. But it may be different if
// the image wasn't copied.
//
// Images with a non-zero origin (Bounds().Min != image.Point{}) are handled
// correctly, and dither identically to the same pixels at the origin, except
// that PixelMapper functions still receive the absolute coordinates.
//
// Dithering an image with empty bounds is a no-op: the image is returned
// unchanged.
//
// The image is validated before any pixel is modified (see validateImage).
// If validation fails the function panics, and the source image is
// guaranteed to be untouched.
//
// Transparency is handled with a fixed policy: fully transparent pixels are
// output as transparent black, their hidden RGB values are never used, and
// they neither generate nor receive diffused error. Partially transparent
// pixels are dithered by their unpremultiplied color and premultiplied again
// on output, with their alpha value preserved. Error that an error diffusion
// matrix would push outside the image bounds is discarded; it is not
// redistributed to other pixels.
func (d *Ditherer) Dither(src image.Image) image.Image {
	if d.invalid() {
		panic("dither: invalid Ditherer")
	}
	if d.Matrix != nil {
		if err := d.Matrix.validate(); err != nil {
			panic("dither: invalid error diffusion matrix: " + err.Error())
		}
	}
	d.validateImage(src)

	if src.Bounds().Empty() {
		// Nothing to do
		return src
	}

	var img draw.Image

	if pi, ok := src.(*image.Paletted); ok {
		if !samePalette(d.palette, pi.Palette) {
			// Can't use this because it will change image colors
			// Instead make a copy, and return that later
			img = copyOfImage(src)
		} else {
			// The image's palette matches the Ditherer's, so it can be
			// safely modified in place.
			img = pi
		}
	} else if img, ok = src.(draw.Image); !ok {
		// Can't be changed
		// Instead make a copy and dither and return that
		img = copyOfImage(src)
	}

	if d.Mapper != nil {
		workers := 1
		if !d.SingleThreaded {
			workers = runtime.GOMAXPROCS(0)
		}
		parallel(workers, img, img, func(x, y int, c color.Color) color.Color {
			r, g, b, a := unpremultAndLinearize(c)

			if a == 0 {
				// Pixel is transparent, it holds no color values
				return color.RGBA64{0, 0, 0, 0}
			}

			return premultAlpha(
				// Use PixelMapper -> find closest palette color -> get that color
				// -> cast to color.RGBA64
				// Comes from d.palette so this cast will always work
				d.palette[d.closestColor(d.Mapper(x, y, r, g, b))].(color.RGBA64),
				a,
			)
		})
		return img
	}

	// Matrix needs to be applied instead

	b := img.Bounds()
	dx, dy := b.Dx(), b.Dy()
	curPx := d.Matrix.CurrentPixel()

	// Store linear values here instead of converting back and forth and storing
	// sRGB values inside the image. The slices are indexed relative to the
	// bounds, so images with a non-zero origin work.
	lins := make([][][3]uint16, dy)
	alphas := make([][]uint16, dy)
	for i := 0; i < len(lins); i++ {
		lins[i] = make([][3]uint16, dx)
		alphas[i] = make([]uint16, dx)
	}

	// Pre-fill those 2D-arrays with the linearized image pixels and alphas
	for yy := 0; yy < dy; yy++ {
		for xx := 0; xx < dx; xx++ {
			r, g, b, a := unpremultAndLinearize(img.At(b.Min.X+xx, b.Min.Y+yy))
			lins[yy][xx] = [3]uint16{r, g, b}
			alphas[yy][xx] = a
		}
	}

	// Now do the actual dithering
	for yy := 0; yy < dy; yy++ {
		y := b.Min.Y + yy
		for xx := 0; xx < dx; xx++ {
			x := b.Min.X + xx

			px := xx
			if d.Serpentine && yy%2 == 0 {
				// Reverse direction
				px = dx - 1 - xx
				x = b.Min.X + px
			}

			if alphas[yy][px] == 0 {
				// Pixel is fully transparent, it holds no color values.
				// Don't dither it, and don't diffuse its hidden RGB values
				// into its neighbors.
				img.Set(x, y, color.RGBA64{0, 0, 0, 0})
				continue
			}

			// Quantize current pixel
			old := lins[yy][px]
			newColorIdx := d.closestColor(old[0], old[1], old[2])
			img.Set(x, y, premultAlpha(d.palette[newColorIdx].(color.RGBA64), alphas[yy][px]))

			new := d.linearPalette[newColorIdx]
			// Quant errors in each channel
			er, eg, eb := int32(old[0])-int32(new[0]), int32(old[1])-int32(new[1]), int32(old[2])-int32(new[2])

			// Diffuse error in two dimensions
			for my := range d.Matrix {
				for mx := range d.Matrix[my] {
					w := d.Matrix[my][mx]
					if w == 0 {
						// Skip, because it won't affect anything
						continue
					}

					// Get the coords of the pixel the error is being applied to
					deltaX, deltaY := d.Matrix.Offset(mx, my, curPx)
					if d.Serpentine && yy%2 == 0 {
						// Reflect the matrix horizontally because we're going right-to-left
						// Otherwise the matrix would change pixels that have already been set
						deltaX *= -1
					}
					tx := px + deltaX
					ty := yy + deltaY

					if tx < 0 || tx >= dx || ty < 0 || ty >= dy {
						// This is outside the image. The error mass that would
						// have gone here is discarded, per the documented
						// policy.
						continue
					}
					if alphas[ty][tx] == 0 {
						// Fully transparent pixels hold no color, so error
						// diffused into them is discarded.
						continue
					}

					t := lins[ty][tx]
					lins[ty][tx] = [3]uint16{
						RoundClamp(float32(t[0]) + float32(er)*w),
						RoundClamp(float32(t[1]) + float32(eg)*w),
						RoundClamp(float32(t[2]) + float32(eb)*w),
					}
				}
			}
		}
	}
	return img
}

// GetColorModel returns a copy of the Ditherer's palette as a color.Model that finds the
// closest color using Euclidean distance in sRGB space.
func (d *Ditherer) GetColorModel() color.Model {
	return color.Palette(copyPalette(d.palette))
}

// DitherConfig is like Dither, but returns an image.Config as well.
func (d *Ditherer) DitherConfig(src draw.Image) (image.Image, image.Config) {
	return d.Dither(src), image.Config{
		ColorModel: d.GetColorModel(),
		Width:      src.Bounds().Dx(),
		Height:     src.Bounds().Dy(),
	}
}

// DitherCopy dithers a copy of the src image and returns it. The src image remains
// unchanged. If you don't need to keep the original image, use Dither.
//
// Like Dither, the image is validated before anything happens, and a failed
// validation panics without touching src.
func (d *Ditherer) DitherCopy(src image.Image) *image.RGBA {
	if d.invalid() {
		panic("dither: invalid Ditherer")
	}
	// Validate before allocating the copy, so a rejected image doesn't
	// trigger a potentially huge allocation.
	d.validateImage(src)

	dst := copyOfImage(src)
	// Can be safely cast because dst is *image.RGBA and .Dither will never need
	// to copy it. And even if it did, it would return this type too.
	return d.Dither(dst).(*image.RGBA)
}

// DitherCopyConfig is like DitherCopy, but returns an image.Config as well.
func (d *Ditherer) DitherCopyConfig(src image.Image) (*image.RGBA, image.Config) {
	return d.DitherCopy(src), image.Config{
		ColorModel: d.GetColorModel(),
		Width:      src.Bounds().Dx(),
		Height:     src.Bounds().Dy(),
	}
}

// DitherPaletted dithers a copy of the src image and returns it as an
// *image.Paletted. The src image remains unchanged. If you don't need an
// *image.Paletted, using Dither or DitherCopy should be preferred.
//
// The palette of the returned image is the same palette the ditherer uses
// internally -- it will be equal to the output of GetPalette().
//
// If the Ditherer's palette has over 256 colors then the function will panic,
// because *image.Paletted does not allow for that.
//
// DitherPaletted can't handle images with transparency.
func (d *Ditherer) DitherPaletted(src image.Image) *image.Paletted {
	if len(d.palette) > 256 {
		panic("dither: DitherPaletted: palette has over 256 colors which *image.Paletted doesn't support")
	}

	rgba := d.DitherCopy(src)
	p := image.NewPaletted(rgba.Bounds(), copyPalette(d.palette))
	copyImage(p, rgba)
	return p
}

// DitherPalettedConfig is like DitherPaletted, but returns an image.Config as well.
//
// DitherPalettedConfig can't handle images with transparency.
func (d *Ditherer) DitherPalettedConfig(src image.Image) (*image.Paletted, image.Config) {
	return d.DitherPaletted(src), image.Config{
		ColorModel: d.GetColorModel(),
		Width:      src.Bounds().Dx(),
		Height:     src.Bounds().Dy(),
	}
}

// RoundClamp clamps the number and rounds it, rounding ties to the nearest even number.
// This should be used if you're writing your own PixelMapper.
func RoundClamp(i float32) uint16 {
	if i < 0 {
		return 0
	}
	if i > 65535 {
		return 65535
	}
	return uint16(math.RoundToEven(float64(i)))
}

// copyImage copies src's pixels into dst.
// They must be the same size.
func copyImage(dst draw.Image, src image.Image) {
	draw.Draw(dst, src.Bounds(), src, src.Bounds().Min, draw.Src)
}

func copyOfImage(img image.Image) *image.RGBA {
	dst := image.NewRGBA(img.Bounds())
	copyImage(dst, img)
	return dst
}

// samePalette returns true if both palettes contain the same colors,
// regardless of order. Colors are compared by their RGBA values, so two
// colors of different concrete types (like color.Gray16{0} and
// color.RGBA64{0, 0, 0, 0xffff}) still count as the same color.
func samePalette(p1 []color.Color, p2 []color.Color) bool {
	if len(p1) != len(p2) {
		return false
	}

	// Modified from: https://stackoverflow.com/a/36000696/7361270

	type rgba struct{ r, g, b, a uint32 }
	key := func(c color.Color) rgba {
		r, g, b, a := c.RGBA()
		return rgba{r, g, b, a}
	}

	diff := make(map[rgba]int, len(p1))
	for _, x := range p1 {
		diff[key(x)]++
	}
	for _, y := range p2 {
		k := key(y)
		// If k is not in diff bail out early
		if _, ok := diff[k]; !ok {
			return false
		}
		diff[k] -= 1
		if diff[k] == 0 {
			delete(diff, k)
		}
	}
	return len(diff) == 0
}
