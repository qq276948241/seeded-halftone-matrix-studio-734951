package dither

// This file contains an independent scalar reference implementation of the
// dithering algorithms, used to verify the optimized library code pixel by
// pixel. It is deliberately written in the most straightforward way possible:
// no helper functions from the library are used, every pixel and channel is
// processed with plain loops, and all intermediate values are kept in simple
// slices. The numeric formulas (sRGB linearization, luminance weights,
// float32 error accumulation, round-to-even clamping) necessarily match the
// documented semantics of the library, but the structure is independent.
//
// The policies implemented here are the documented ones:
//   - All math happens in linear RGB space, as 16-bit values.
//   - The closest palette color uses luminance-weighted Euclidean distance;
//     ties are won by the palette color that comes first.
//   - Fully transparent pixels hold no color: they are output as transparent
//     black, they are never dithered, and they neither generate nor receive
//     diffused error.
//   - Partially transparent pixels are dithered by their unpremultiplied
//     color and premultiplied again on output; alpha is preserved.
//   - Error that would be diffused outside the image bounds is discarded.

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// refLinearize1 applies the sRGB EOTF to a single value in [0, 1].
func refLinearize1(v float64) float64 {
	if v <= 0.04045 {
		return v / 12.92
	}
	return math.Pow((v+0.055)/1.055, 2.4)
}

// refLin16 linearizes a 16-bit sRGB channel value.
func refLin16(v uint16) uint16 {
	return uint16(math.RoundToEven(refLinearize1(float64(v)/65535.0) * 65535.0))
}

// refUnpack converts any color to unpremultiplied linear RGB plus alpha.
func refUnpack(c color.Color) (uint16, uint16, uint16, uint16) {
	n := color.NRGBA64Model.Convert(c).(color.NRGBA64)
	return refLin16(n.R), refLin16(n.G), refLin16(n.B), n.A
}

// refSqDiff is the squared difference of two channel values, shifted right
// by two to avoid overflow in later sums.
func refSqDiff(a, b uint16) uint32 {
	d := uint32(a) - uint32(b)
	return (d * d) >> 2
}

// refLinearPalette converts palette colors to linear RGB triples.
func refLinearPalette(pal []color.Color) [][3]uint16 {
	out := make([][3]uint16, len(pal))
	for i, c := range pal {
		r, g, b, _ := refUnpack(c)
		out[i] = [3]uint16{r, g, b}
	}
	return out
}

// refOpaqueRGBA64 converts a palette color to its RGBA64 form, the same form
// the Ditherer stores its palette in.
func refOpaqueRGBA64(c color.Color) color.RGBA64 {
	r, g, b, a := c.RGBA()
	return color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)}
}

// refClosest returns the index of the closest palette color, using
// luminance-weighted Euclidean distance in linear RGB. Ties are won by the
// lowest index.
func refClosest(linPal [][3]uint16, r, g, b uint16) int {
	best := uint32(math.MaxUint32)
	idx := 0
	for i, c := range linPal {
		// Weights 0.2126, 0.7152, 0.0722 as fractions, in integer math.
		dist := uint32(
			1063*uint64(refSqDiff(r, c[0]))/5000 +
				447*uint64(refSqDiff(g, c[1]))/625 +
				361*uint64(refSqDiff(b, c[2]))/5000,
		)
		if dist < best {
			if dist == 0 {
				return i
			}
			idx, best = i, dist
		}
	}
	return idx
}

// refRoundClamp rounds ties to even and clamps to [0, 65535].
func refRoundClamp(v float32) uint16 {
	if v < 0 {
		return 0
	}
	if v > 65535 {
		return 65535
	}
	return uint16(math.RoundToEven(float64(v)))
}

// refPremult premultiplies an sRGB color by the given alpha, following the
// documented alpha policy.
func refPremult(c color.RGBA64, a uint16) color.RGBA64 {
	if a == 0 {
		return color.RGBA64{}
	}
	if a == 0xffff {
		return c
	}
	return color.RGBA64{
		R: uint16(uint32(c.R) * uint32(a) / 0xffff),
		G: uint16(uint32(c.G) * uint32(a) / 0xffff),
		B: uint16(uint32(c.B) * uint32(a) / 0xffff),
		A: a,
	}
}

// refOrdered applies an ordered dithering matrix (already converted to
// additive values) and quantizes each pixel to the palette.
func refOrdered(src image.Image, pal []color.Color, add [][]float32) *image.RGBA64 {
	b := src.Bounds()
	dst := image.NewRGBA64(b)
	linPal := refLinearPalette(pal)
	ydim := len(add)
	xdim := len(add[0])
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bb, a := refUnpack(src.At(x, y))
			if a == 0 {
				dst.SetRGBA64(x, y, color.RGBA64{})
				continue
			}
			delta := add[y%ydim][x%xdim]
			idx := refClosest(linPal,
				refRoundClamp(float32(r)+delta),
				refRoundClamp(float32(g)+delta),
				refRoundClamp(float32(bb)+delta),
			)
			dst.SetRGBA64(x, y, refPremult(refOpaqueRGBA64(pal[idx]), a))
		}
	}
	return dst
}

// refCurrentPixel finds the current pixel of an error diffusion matrix: the
// right-most zero in the top row.
func refCurrentPixel(m [][]float32) int {
	for i, v := range m[0] {
		if v != 0 {
			return i - 1
		}
	}
	return len(m[0]) / 2
}

// refDiffuse applies error diffusion dithering with the documented policies.
func refDiffuse(src image.Image, pal []color.Color, m [][]float32, serpentine bool) *image.RGBA64 {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewRGBA64(b)
	linPal := refLinearPalette(pal)

	val := make([][][3]uint16, h)
	alpha := make([][]uint16, h)
	for yy := 0; yy < h; yy++ {
		val[yy] = make([][3]uint16, w)
		alpha[yy] = make([]uint16, w)
		for xx := 0; xx < w; xx++ {
			r, g, bb, a := refUnpack(src.At(b.Min.X+xx, b.Min.Y+yy))
			val[yy][xx] = [3]uint16{r, g, bb}
			alpha[yy][xx] = a
		}
	}

	curPx := refCurrentPixel(m)

	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			px := xx
			if serpentine && yy%2 == 0 {
				px = w - 1 - xx
			}

			if alpha[yy][px] == 0 {
				// Fully transparent: no color, no error.
				dst.SetRGBA64(b.Min.X+px, b.Min.Y+yy, color.RGBA64{})
				continue
			}

			old := val[yy][px]
			idx := refClosest(linPal, old[0], old[1], old[2])
			dst.SetRGBA64(b.Min.X+px, b.Min.Y+yy,
				refPremult(refOpaqueRGBA64(pal[idx]), alpha[yy][px]))

			nw := linPal[idx]
			er := int32(old[0]) - int32(nw[0])
			eg := int32(old[1]) - int32(nw[1])
			eb := int32(old[2]) - int32(nw[2])

			for my := range m {
				for mx := range m[my] {
					weight := m[my][mx]
					if weight == 0 {
						continue
					}
					dxOff := mx - curPx
					if serpentine && yy%2 == 0 {
						dxOff = -dxOff
					}
					tx, ty := px+dxOff, yy+my
					if tx < 0 || tx >= w || ty < 0 || ty >= h {
						// Error mass off the edge is discarded.
						continue
					}
					if alpha[ty][tx] == 0 {
						// Transparent pixels receive no error.
						continue
					}
					t := val[ty][tx]
					val[ty][tx] = [3]uint16{
						refRoundClamp(float32(t[0]) + float32(er)*weight),
						refRoundClamp(float32(t[1]) + float32(eg)*weight),
						refRoundClamp(float32(t[2]) + float32(eb)*weight),
					}
				}
			}
		}
	}
	return dst
}

// refAdditiveMatrix converts a threshold matrix to additive values, the same
// conversion the library documents for ordered dithering.
func refAdditiveMatrix(matrix [][]uint, max uint, strength float32) [][]float32 {
	out := make([][]float32, len(matrix))
	for i := range matrix {
		out[i] = make([]float32, len(matrix[i]))
		for j, v := range matrix[i] {
			out[i][j] = 65535.0 * strength * (float32(v+1.0)/float32(max) - 0.50000006)
		}
	}
	return out
}

// makeRGBA64 builds a test image with the given bounds and pixel function.
func makeRGBA64(min image.Point, w, h int, fn func(x, y int) color.RGBA64) *image.RGBA64 {
	img := image.NewRGBA64(image.Rect(min.X, min.Y, min.X+w, min.Y+h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA64(min.X+x, min.Y+y, fn(x, y))
		}
	}
	return img
}

// cloneRGBA64 returns a deep copy of the image as *image.RGBA64, preserving
// the bounds (including a non-zero origin).
func cloneRGBA64(src image.Image) *image.RGBA64 {
	b := src.Bounds()
	dst := image.NewRGBA64(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x, y, src.At(x, y))
		}
	}
	return dst
}

// ditherRGBA64Exact dithers a clone of src in place, so the result keeps
// 16-bit precision and can be compared exactly with the reference
// implementation. (DitherCopy always returns an 8-bit *image.RGBA, which
// would lose precision for partially transparent pixels.)
func ditherRGBA64Exact(d *Ditherer, src image.Image) image.Image {
	return d.Dither(cloneRGBA64(src))
}

// variedColor returns an opaque pseudo-random-but-deterministic color.
func variedColor(x, y int) color.RGBA64 {
	return color.RGBA64{
		R: uint16((x*9377 + y*1499 + 12345) % 65536),
		G: uint16((x*3011 + y*7919 + 777) % 65536),
		B: uint16((x*577 + y*10499 + 40000) % 65536),
		A: 0xffff,
	}
}

// variedColorAlpha is like variedColor, but some pixels are partially
// transparent (with valid premultiplied RGB), and some are fully transparent
// with hidden RGB values that must never be used.
func variedColorAlpha(x, y int) color.RGBA64 {
	c := variedColor(x, y)
	switch (x + y*3) % 7 {
	case 0:
		// Fully transparent, with non-zero hidden RGB on purpose
		c.A = 0
	case 3:
		c.A = 0x8000
		c.R %= 0x8001
		c.G %= 0x8001
		c.B %= 0x8001
	case 5:
		c.A = 0x1234
		c.R %= 0x1235
		c.G %= 0x1235
		c.B %= 0x1235
	}
	return c
}

// assertSamePixels fails the test if the two images differ in any channel of
// any pixel. Bounds must be equal.
func assertSamePixels(t *testing.T, want, got image.Image) {
	t.Helper()
	if !want.Bounds().Eq(got.Bounds()) {
		t.Fatalf("bounds differ: want %v, got %v", want.Bounds(), got.Bounds())
	}
	b := want.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			wr, wg, wb, wa := want.At(x, y).RGBA()
			gr, gg, gb, ga := got.At(x, y).RGBA()
			if wr != gr || wg != gg || wb != gb || wa != ga {
				t.Fatalf("pixel (%d, %d) differs: want (%d, %d, %d, %d), got (%d, %d, %d, %d)",
					x, y, wr, wg, wb, wa, gr, gg, gb, ga)
			}
		}
	}
}

var refPalettes = map[string][]color.Color{
	"blackWhite": {color.Black, color.White},
	"rgb": {
		color.RGBA{255, 0, 0, 255},
		color.RGBA{0, 255, 0, 255},
		color.RGBA{0, 0, 255, 255},
		color.Black,
		color.White,
	},
	"grays": {
		color.Gray{0},
		color.Gray{85},
		color.Gray{170},
		color.Gray{255},
	},
}

func TestReferenceOrderedBayer(t *testing.T) {
	img := makeRGBA64(image.Point{}, 9, 7, variedColor)
	imgAlpha := makeRGBA64(image.Point{}, 9, 7, variedColorAlpha)
	imgOffset := makeRGBA64(image.Point{X: 5, Y: 3}, 9, 7, variedColor)

	for name, pal := range refPalettes {
		for _, size := range [][2]uint{{2, 2}, {4, 4}, {8, 4}} {
			matrix := bayerMatrix(size[0], size[1])
			add := refAdditiveMatrix(matrix, size[0]*size[1], 1.0)

			d := NewDitherer(pal)
			d.Mapper = Bayer(size[0], size[1], 1.0)

			for _, src := range []*image.RGBA64{img, imgAlpha, imgOffset} {
				want := refOrdered(src, pal, add)
				got := ditherRGBA64Exact(d, src)
				if !sameImage(want, got) {
					t.Errorf("palette %s, bayer %dx%d: mismatch", name, size[0], size[1])
					assertSamePixels(t, want, got)
				}
			}
		}
	}
}

func TestReferenceOrderedCustomMatrix(t *testing.T) {
	img := makeRGBA64(image.Point{}, 11, 6, variedColor)
	imgAlpha := makeRGBA64(image.Point{}, 11, 6, variedColorAlpha)

	odms := []OrderedDitherMatrix{
		ClusteredDot4x4,
		ClusteredDotDiagonal8x8,
		{Matrix: [][]uint{{0, 2}, {3, 1}}, Max: 4},
	}
	for _, odm := range odms {
		for _, strength := range []float32{1.0, 0.7} {
			add := refAdditiveMatrix(odm.Matrix, odm.Max, strength)
			for name, pal := range refPalettes {
				d := NewDitherer(pal)
				d.Mapper = PixelMapperFromMatrix(odm, strength)
				for _, src := range []*image.RGBA64{img, imgAlpha} {
					want := refOrdered(src, pal, add)
					got := ditherRGBA64Exact(d, src)
					if !sameImage(want, got) {
						t.Errorf("palette %s, strength %v: mismatch", name, strength)
						assertSamePixels(t, want, got)
					}
				}
			}
		}
	}
}

func TestReferenceDiffusion(t *testing.T) {
	imgs := []*image.RGBA64{
		makeRGBA64(image.Point{}, 7, 5, variedColor),
		makeRGBA64(image.Point{}, 7, 5, variedColorAlpha),
		makeRGBA64(image.Point{X: 4, Y: 9}, 6, 4, variedColor),
		makeRGBA64(image.Point{}, 1, 1, variedColor),
		makeRGBA64(image.Point{}, 1, 6, variedColor),
		makeRGBA64(image.Point{}, 6, 1, variedColor),
	}

	matrices := map[string]ErrorDiffusionMatrix{
		"simple2d":   Simple2D,
		"floydStein": FloydSteinberg,
		"atkinson":   Atkinson,
		"jjn":        JarvisJudiceNinke,
		"sierraLite": SierraLite,
		"fsHalf":     ErrorDiffusionStrength(FloydSteinberg, 0.5),
	}

	for mname, m := range matrices {
		for _, serpentine := range []bool{false, true} {
			for pname, pal := range refPalettes {
				d := NewDitherer(pal)
				d.Matrix = m
				d.Serpentine = serpentine
				for i, src := range imgs {
					want := refDiffuse(src, pal, m, serpentine)
					got := ditherRGBA64Exact(d, src)
					if !sameImage(want, got) {
						t.Errorf("matrix %s serpentine=%v palette %s image %d: mismatch",
							mname, serpentine, pname, i)
						assertSamePixels(t, want, got)
					}
				}
			}
		}
	}
}

func TestReferenceDiffusionNonZeroOriginMatchesZeroOrigin(t *testing.T) {
	// The matrix path must dither a shifted image identically to the same
	// pixels at the origin.
	fn := variedColor
	atOrigin := makeRGBA64(image.Point{}, 8, 6, fn)
	shifted := makeRGBA64(image.Point{X: 13, Y: 7}, 8, 6, fn)

	for _, serpentine := range []bool{false, true} {
		d := NewDitherer(blackWhite)
		d.Matrix = FloydSteinberg
		d.Serpentine = serpentine

		gotOrigin := d.DitherCopy(atOrigin)
		gotShifted := d.DitherCopy(shifted)

		b := gotOrigin.Bounds()
		for y := 0; y < b.Dy(); y++ {
			for x := 0; x < b.Dx(); x++ {
				if !sameColor(gotOrigin.At(x, y), gotShifted.At(13+x, 7+y)) {
					t.Fatalf("serpentine=%v: pixel (%d, %d) differs between origin and shifted image",
						serpentine, x, y)
				}
			}
		}
	}
}

func TestClosestColorTieBreak(t *testing.T) {
	d := NewDitherer([]color.Color{color.Black, color.White})
	// Craft a linear palette where an exact midpoint exists.
	d.linearPalette = [][3]uint16{
		{100, 100, 100},
		{200, 200, 200},
	}
	// 150 is exactly between 100 and 200; the first palette color must win.
	if got := d.closestColor(150, 150, 150); got != 0 {
		t.Errorf("tie should be won by first palette color, got index %d", got)
	}
	// Swapping the palette must swap the winner.
	d.linearPalette = [][3]uint16{
		{200, 200, 200},
		{100, 100, 100},
	}
	if got := d.closestColor(150, 150, 150); got != 0 {
		t.Errorf("tie should be won by first palette color after swap, got index %d", got)
	}
	// Exact match short-circuits.
	if got := d.closestColor(200, 200, 200); got != 0 {
		t.Errorf("exact match should return its index, got %d", got)
	}
}

func TestRoundClamp(t *testing.T) {
	cases := []struct {
		in   float32
		want uint16
	}{
		{-1, 0},
		{-0.4, 0},
		{0, 0},
		{0.5, 0}, // ties to even
		{1.5, 2}, // ties to even
		{2.5, 2}, // ties to even
		{3.5, 4}, // ties to even
		{100.4, 100},
		{65535, 65535},
		{65535.4, 65535},
		{65536, 65535},
		{1e9, 65535},
	}
	for _, c := range cases {
		if got := RoundClamp(c.in); got != c.want {
			t.Errorf("RoundClamp(%v) = %d, want %d", c.in, got, c.want)
		}
		if got := refRoundClamp(c.in); got != c.want {
			t.Errorf("refRoundClamp(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}
