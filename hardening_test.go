package dither

// Tests for the input validation, determinism, transparency, and concurrency
// guarantees of the package. These tests do not rely on any golden images;
// they construct their inputs programmatically and check exact pixels or
// exact panics.

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"sync"
	"testing"
)

// assertPanics fails the test if f does not panic.
func assertPanics(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("expected panic: %s", what)
		}
	}()
	f()
}

// grayRamp returns an opaque grayscale test image with the given bounds.
func grayRamp(min image.Point, w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(min.X, min.Y, min.X+w, min.Y+h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8((x*23 + y*41) % 256)
			img.SetRGBA(min.X+x, min.Y+y, color.RGBA{v, v, v, 255})
		}
	}
	return img
}

func TestDitherInvalidConfigs(t *testing.T) {
	img := grayRamp(image.Point{}, 4, 4)

	// Nothing set
	d := NewDitherer(blackWhite)
	assertPanics(t, "no Matrix, Mapper or Special set", func() { d.Dither(img) })

	// More than one set
	d = NewDitherer(blackWhite)
	d.Matrix = FloydSteinberg
	d.Mapper = Bayer(2, 2, 1)
	assertPanics(t, "Matrix and Mapper both set", func() { d.Dither(img) })

	d = NewDitherer(blackWhite)
	d.Mapper = Bayer(2, 2, 1)
	d.Special = 1
	assertPanics(t, "Mapper and Special both set", func() { d.Dither(img) })

	// Special is not supported
	d = NewDitherer(blackWhite)
	d.Special = 1
	assertPanics(t, "Special set", func() { d.Dither(img) })

	// Same for DitherCopy
	d = NewDitherer(blackWhite)
	assertPanics(t, "DitherCopy with invalid Ditherer", func() { d.DitherCopy(img) })
}

func TestNewDithererEmptyPalette(t *testing.T) {
	if NewDitherer(nil) != nil {
		t.Error("nil palette should return nil Ditherer")
	}
	if NewDitherer([]color.Color{}) != nil {
		t.Error("empty palette should return nil Ditherer")
	}
}

func TestDitherNilAndEmptyImage(t *testing.T) {
	d := NewDitherer(blackWhite)
	d.Matrix = FloydSteinberg

	assertPanics(t, "nil image", func() { d.Dither(nil) })

	// Empty images are a no-op and must not panic, for both paths.
	empty := image.NewRGBA(image.Rectangle{})
	if got := d.Dither(empty); got != image.Image(empty) {
		t.Error("empty image should be returned unchanged")
	}
	emptyOffset := image.NewRGBA(image.Rect(5, 5, 5, 5))
	if got := d.Dither(emptyOffset); got != image.Image(emptyOffset) {
		t.Error("empty image with non-zero origin should be returned unchanged")
	}

	d2 := NewDitherer(blackWhite)
	d2.Mapper = Bayer(2, 2, 1)
	if got := d2.Dither(image.NewRGBA(image.Rectangle{})); got == nil {
		t.Error("empty image should be returned, not nil")
	}
	if got := d2.DitherCopy(image.NewRGBA(image.Rectangle{})); got == nil {
		t.Error("DitherCopy of empty image should return an image, not nil")
	}
}

func TestPixelBudget(t *testing.T) {
	src := grayRamp(image.Point{}, 10, 10) // 100 pixels
	before := grayRamp(image.Point{}, 10, 10)
	copy(before.Pix, src.Pix)

	d := NewDitherer(blackWhite)
	d.Matrix = FloydSteinberg
	d.MaxPixels = 99

	assertPanics(t, "image over pixel budget", func() { d.Dither(src) })
	if !sameImage(before, src) {
		t.Error("source image was modified despite budget rejection")
	}

	assertPanics(t, "DitherCopy over pixel budget", func() { d.DitherCopy(src) })
	if !sameImage(before, src) {
		t.Error("source image was modified despite budget rejection (DitherCopy)")
	}

	// Exactly at the budget is fine.
	d.MaxPixels = 100
	_ = d.Dither(src)

	// Negative disables the budget.
	src2 := grayRamp(image.Point{}, 10, 10)
	d.MaxPixels = -1
	_ = d.Dither(src2)

	// Zero means the default budget, which normal images pass.
	src3 := grayRamp(image.Point{}, 10, 10)
	d.MaxPixels = 0
	_ = d.Dither(src3)
}

func TestStrideValidation(t *testing.T) {
	d := NewDitherer(blackWhite)
	d.Matrix = FloydSteinberg

	// A corrupt *image.RGBA: bounds say 10x10 (needs >= 400 bytes with
	// stride 40) but the Pix slice is far too small.
	corrupt := &image.RGBA{
		Pix:    make([]uint8, 40),
		Stride: 40,
		Rect:   image.Rect(0, 0, 10, 10),
	}
	assertPanics(t, "corrupt stride", func() { d.Dither(corrupt) })

	// A corrupt stride that's too small for one row.
	corrupt2 := &image.RGBA{
		Pix:    make([]uint8, 400),
		Stride: 4, // needs at least 4*10 = 40
		Rect:   image.Rect(0, 0, 10, 10),
	}
	assertPanics(t, "stride smaller than one row", func() { d.Dither(corrupt2) })

	// A legitimate sub-image with a larger stride than its width must work.
	parent := grayRamp(image.Point{}, 20, 20)
	sub := parent.SubImage(image.Rect(5, 5, 15, 12)).(*image.RGBA)
	_ = d.Dither(sub)
}

func TestMatrixValidation(t *testing.T) {
	newImg := func() *image.RGBA { return grayRamp(image.Point{}, 6, 6) }

	bad := map[string]ErrorDiffusionMatrix{
		"empty":          {},
		"emptyFirstRow":  {{}},
		"ragged":         {{0, 0, 0.5}, {0.5, 0.5}},
		"noCurrentPixel": {{0.5, 0}}, // top row starts with non-zero
		"modifiesPast":   {{0.5, 0, 0.25}, {0.25, 0.25, 0}},
		"nan":            {{0, 0, float32(math.NaN())}, {0.5, 0.5, 0}},
		"inf":            {{0, 0, float32(math.Inf(1))}, {0.5, 0.5, 0}},
	}
	for name, m := range bad {
		d := NewDitherer(blackWhite)
		d.Matrix = m
		src := newImg()
		before := newImg()
		copy(before.Pix, src.Pix)
		assertPanics(t, "invalid matrix: "+name, func() { d.Dither(src) })
		if !sameImage(before, src) {
			t.Errorf("matrix %s: source image was modified despite rejection", name)
		}
	}

	// A valid custom matrix works.
	d := NewDitherer(blackWhite)
	d.Matrix = ErrorDiffusionMatrix{
		{0, 0, 0.5},
		{0.25, 0.25, 0},
	}
	_ = d.Dither(newImg())
}

func TestErrorDiffusionStrengthCopiesAndValidates(t *testing.T) {
	// Even with strength 1.0 the matrix must be copied, so that the caller
	// can't mutate the shared matrix while another goroutine dithers with it.
	m := ErrorDiffusionStrength(FloydSteinberg, 1.0)
	m[0][2] = 999
	if FloydSteinberg[0][2] == 999 {
		t.Error("ErrorDiffusionStrength returned the shared matrix, it must copy")
	}

	assertPanics(t, "NaN strength", func() {
		ErrorDiffusionStrength(FloydSteinberg, float32(math.NaN()))
	})
	assertPanics(t, "Inf strength", func() {
		ErrorDiffusionStrength(FloydSteinberg, float32(math.Inf(1)))
	})
}

func TestBayerValidation(t *testing.T) {
	assertPanics(t, "Bayer x=0", func() { Bayer(0, 2, 1) })
	assertPanics(t, "Bayer y=0", func() { Bayer(2, 0, 1) })
	assertPanics(t, "Bayer non-power-of-two", func() { Bayer(6, 4, 1) })
	assertPanics(t, "Bayer NaN strength", func() { Bayer(2, 2, float32(math.NaN())) })
	assertPanics(t, "Bayer Inf strength", func() { Bayer(2, 2, float32(math.Inf(-1))) })

	// Documented exceptions and regular sizes work.
	_ = Bayer(3, 3, 1)
	_ = Bayer(5, 3, 1)
	_ = Bayer(3, 5, 1)
	_ = Bayer(16, 8, 0.5)
	_ = Bayer(1, 1, 1)
}

func TestPixelMapperFromMatrixValidation(t *testing.T) {
	assertPanics(t, "empty matrix", func() {
		PixelMapperFromMatrix(OrderedDitherMatrix{Matrix: [][]uint{}, Max: 4}, 1)
	})
	assertPanics(t, "empty row", func() {
		PixelMapperFromMatrix(OrderedDitherMatrix{Matrix: [][]uint{{}}, Max: 4}, 1)
	})
	assertPanics(t, "Max zero", func() {
		PixelMapperFromMatrix(OrderedDitherMatrix{Matrix: [][]uint{{0, 1}, {2, 3}}, Max: 0}, 1)
	})
	assertPanics(t, "ragged", func() {
		PixelMapperFromMatrix(OrderedDitherMatrix{Matrix: [][]uint{{0, 1}, {2}}, Max: 4}, 1)
	})
	assertPanics(t, "value >= Max", func() {
		PixelMapperFromMatrix(OrderedDitherMatrix{Matrix: [][]uint{{0, 4}, {2, 3}}, Max: 4}, 1)
	})
	assertPanics(t, "NaN strength", func() {
		PixelMapperFromMatrix(OrderedDitherMatrix{Matrix: [][]uint{{0, 1}, {2, 3}}, Max: 4}, float32(math.NaN()))
	})
}

func TestRandomNoiseValidation(t *testing.T) {
	assertPanics(t, "min >= max", func() { RandomNoiseGrayscale(1, 0.5, -0.5) })
	assertPanics(t, "min == max", func() { RandomNoiseGrayscale(1, 0.5, 0.5) })
	assertPanics(t, "NaN", func() { RandomNoiseGrayscale(1, float32(math.NaN()), 0.5) })
	assertPanics(t, "RGB min >= max", func() {
		RandomNoiseRGB(1, -0.5, 0.5, 0.6, 0.5, -0.5, 0.5)
	})
}

func TestRandomNoiseDeterministic(t *testing.T) {
	newDitherer := func(seed int64) *Ditherer {
		d := NewDitherer(blackWhite)
		d.Mapper = RandomNoiseGrayscale(seed, -0.5, 0.5)
		return d
	}

	// Same seed: identical output, whether single- or multi-threaded.
	a := newDitherer(42).Dither(grayRamp(image.Point{}, 32, 32))
	b := newDitherer(42).Dither(grayRamp(image.Point{}, 32, 32))
	if !sameImage(a, b) {
		t.Error("same seed produced different output")
	}

	d := newDitherer(42)
	d.SingleThreaded = true
	c := d.Dither(grayRamp(image.Point{}, 32, 32))
	if !sameImage(a, c) {
		t.Error("single-threaded and multi-threaded output differ")
	}

	// Different seed: output must differ somewhere.
	e := newDitherer(43).Dither(grayRamp(image.Point{}, 32, 32))
	if sameImage(a, e) {
		t.Error("different seeds produced identical output")
	}
}

func TestTransparentHiddenRGBNotUsed(t *testing.T) {
	// Two images identical except for the RGB values hidden inside fully
	// transparent pixels. The dithered output must be identical, for both
	// the mapper and the matrix path.
	mk := func(hidden color.NRGBA) *image.NRGBA {
		img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
		for y := 0; y < 8; y++ {
			for x := 0; x < 8; x++ {
				if (x+y)%3 == 0 {
					img.SetNRGBA(x, y, hidden) // transparent, hidden RGB
				} else {
					v := uint8((x*31 + y*17) % 256)
					img.SetNRGBA(x, y, color.NRGBA{v, v, v, 255})
				}
			}
		}
		return img
	}
	img1 := mk(color.NRGBA{255, 0, 0, 0}) // hidden red
	img2 := mk(color.NRGBA{0, 255, 0, 0}) // hidden green

	for _, path := range []string{"mapper", "matrix"} {
		newD := func() *Ditherer {
			d := NewDitherer(blackWhite)
			if path == "mapper" {
				d.Mapper = Bayer(4, 4, 1)
			} else {
				d.Matrix = FloydSteinberg
			}
			return d
		}
		got1 := newD().Dither(img1)
		got2 := newD().Dither(img2)
		if !sameImage(got1, got2) {
			t.Errorf("%s path: hidden RGB of transparent pixels affected the output", path)
		}

		// Fully transparent pixels must come out as transparent black.
		b := got1.Bounds()
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if (x+y)%3 == 0 {
					r, g, bl, a := got1.At(x, y).RGBA()
					if r != 0 || g != 0 || bl != 0 || a != 0 {
						t.Errorf("%s path: transparent pixel (%d,%d) = (%d,%d,%d,%d), want (0,0,0,0)",
							path, x, y, r, g, bl, a)
					}
				}
			}
		}
	}
}

func TestPartialAlphaPreserved(t *testing.T) {
	img := image.NewNRGBA64(image.Rect(0, 0, 6, 6))
	alphas := map[image.Point]uint16{}
	for y := 0; y < 6; y++ {
		for x := 0; x < 6; x++ {
			a := uint16((x*9000 + y*13000) % 65536)
			img.SetNRGBA64(x, y, color.NRGBA64{
				R: uint16((x*31 + y*17) % 256 * 257),
				G: 0x8080,
				B: 0x2020,
				A: a,
			})
			alphas[image.Point{x, y}] = a
		}
	}

	for _, path := range []string{"mapper", "matrix"} {
		d := NewDitherer(blackWhite)
		if path == "mapper" {
			d.Mapper = Bayer(2, 2, 1)
		} else {
			d.Matrix = FloydSteinberg
		}
		got := d.Dither(cloneRGBA64(img))
		for y := 0; y < 6; y++ {
			for x := 0; x < 6; x++ {
				_, _, _, a := got.At(x, y).RGBA()
				want := uint32(alphas[image.Point{x, y}])
				// The output is premultiplied; the alpha itself must be
				// exactly the original alpha.
				if a != want {
					t.Errorf("%s path: pixel (%d,%d) alpha = %d, want %d", path, x, y, a, want)
				}
			}
		}
	}
}

func TestSourceUnchanged(t *testing.T) {
	d := NewDitherer(blackWhite)
	d.Matrix = FloydSteinberg

	// DitherCopy must never touch src, including for images with alpha and
	// hidden RGB in transparent pixels.
	src := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			src.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x * 30), G: uint8(y * 30), B: 200,
				A: uint8((x * y * 37) % 256),
			})
		}
	}
	before := image.NewNRGBA(src.Bounds())
	copy(before.Pix, src.Pix)

	_ = d.DitherCopy(src)
	if !sameImage(before, src) {
		t.Error("DitherCopy modified the source image")
	}

	// Dither on an image that isn't a draw.Image must copy and leave src
	// alone. *image.YCbCr has no Set method.
	ycbcr := image.NewYCbCr(image.Rect(0, 0, 8, 8), image.YCbCrSubsampleRatio420)
	for i := range ycbcr.Y {
		ycbcr.Y[i] = uint8(i * 3)
	}
	yBefore := make([]uint8, len(ycbcr.Y))
	copy(yBefore, ycbcr.Y)
	out := d.Dither(ycbcr)
	for i := range ycbcr.Y {
		if ycbcr.Y[i] != yBefore[i] {
			t.Fatal("Dither modified a non-draw.Image source")
		}
	}
	if out == image.Image(ycbcr) {
		t.Error("Dither should have returned a copy for a non-draw.Image")
	}

	// Dither on a paletted image with a different palette must copy.
	pal := image.NewPaletted(image.Rect(0, 0, 4, 4), color.Palette{
		color.RGBA{1, 2, 3, 255}, color.RGBA{4, 5, 6, 255},
	})
	pixBefore := make([]uint8, len(pal.Pix))
	copy(pixBefore, pal.Pix)
	_ = d.Dither(pal)
	for i := range pal.Pix {
		if pal.Pix[i] != pixBefore[i] {
			t.Fatal("Dither modified a paletted image with a different palette")
		}
	}
}

func TestPalettedSamePaletteInPlace(t *testing.T) {
	// Regression test: a paletted image whose palette matches the Ditherer's
	// must be dithered in place (this used to panic on a nil image).
	d := NewDitherer(blackWhite)
	d.Matrix = FloydSteinberg

	pal := image.NewPaletted(image.Rect(0, 0, 8, 8), color.Palette{color.Black, color.White})
	for i := range pal.Pix {
		pal.Pix[i] = uint8(i % 2)
	}
	got := d.Dither(pal)
	if got != image.Image(pal) {
		t.Error("expected in-place dithering of paletted image with same palette")
	}

	// Palette order doesn't matter for the comparison.
	pal2 := image.NewPaletted(image.Rect(0, 0, 8, 8), color.Palette{color.White, color.Black})
	for i := range pal2.Pix {
		pal2.Pix[i] = uint8(i % 2)
	}
	got2 := d.Dither(pal2)
	if got2 != image.Image(pal2) {
		t.Error("expected in-place dithering of paletted image with same palette, different order")
	}
}

func TestRepeatedCallsDifferentSizes(t *testing.T) {
	// Dithering images of different sizes one after another must not leave
	// any residue from previous runs.
	d := NewDitherer(redGreenBlack)
	d.Matrix = JarvisJudiceNinke

	first := d.DitherCopy(grayRamp(image.Point{}, 5, 5))
	_ = d.DitherCopy(grayRamp(image.Point{}, 17, 3))
	_ = d.DitherCopy(grayRamp(image.Point{}, 2, 29))
	second := d.DitherCopy(grayRamp(image.Point{}, 5, 5))
	if !sameImage(first, second) {
		t.Error("same image dithered before and after other sizes differs")
	}

	dm := NewDitherer(redGreenBlack)
	dm.Mapper = Bayer(4, 4, 0.8)
	firstM := dm.DitherCopy(grayRamp(image.Point{}, 5, 5))
	_ = dm.DitherCopy(grayRamp(image.Point{}, 17, 3))
	secondM := dm.DitherCopy(grayRamp(image.Point{}, 5, 5))
	if !sameImage(firstM, secondM) {
		t.Error("mapper path: same image dithered before and after other sizes differs")
	}
}

func TestConcurrentDither(t *testing.T) {
	// The same Ditherer used from many goroutines at once must produce the
	// same output as sequential use, with no data races (run with -race).
	mappers := map[string]PixelMapper{
		"bayer":     Bayer(4, 4, 1),
		"random":    RandomNoiseGrayscale(7, -0.5, 0.5),
		"randomRGB": RandomNoiseRGB(7, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
	}
	for name, mapper := range mappers {
		d := NewDitherer(redGreenYellowBlack)
		d.Mapper = mapper

		want := d.DitherCopy(grayRamp(image.Point{}, 24, 24))

		const n = 8
		results := make([]image.Image, n)
		var wg sync.WaitGroup
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				results[i] = d.DitherCopy(grayRamp(image.Point{}, 24, 24))
			}(i)
		}
		wg.Wait()
		for i, got := range results {
			if !sameImage(want, got) {
				t.Errorf("mapper %s: concurrent result %d differs from sequential", name, i)
			}
		}
	}

	// Matrix path concurrently on separate images.
	d := NewDitherer(blackWhite)
	d.Matrix = FloydSteinberg
	want := d.DitherCopy(grayRamp(image.Point{}, 16, 16))
	var wg sync.WaitGroup
	errs := make(chan bool, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if !sameImage(want, d.DitherCopy(grayRamp(image.Point{}, 16, 16))) {
				errs <- true
			}
		}()
	}
	wg.Wait()
	close(errs)
	for range errs {
		t.Error("matrix path: concurrent result differs from sequential")
	}
}

func TestInputTypesConsistent(t *testing.T) {
	// The same opaque image stored as different standard types must dither
	// to the same pixels.
	mk := func() []image.Image {
		r := image.Rect(0, 0, 8, 8)
		rgba := image.NewRGBA(r)
		nrgba := image.NewNRGBA(r)
		rgba64 := image.NewRGBA64(r)
		nrgba64 := image.NewNRGBA64(r)
		gray := image.NewGray(r)
		gray16 := image.NewGray16(r)
		for y := 0; y < 8; y++ {
			for x := 0; x < 8; x++ {
				v := uint8((x*37 + y*53) % 256)
				rgba.SetRGBA(x, y, color.RGBA{v, v, v, 255})
				nrgba.SetNRGBA(x, y, color.NRGBA{v, v, v, 255})
				rgba64.SetRGBA64(x, y, color.RGBA64{uint16(v) * 257, uint16(v) * 257, uint16(v) * 257, 0xffff})
				nrgba64.SetNRGBA64(x, y, color.NRGBA64{uint16(v) * 257, uint16(v) * 257, uint16(v) * 257, 0xffff})
				gray.SetGray(x, y, color.Gray{v})
				gray16.SetGray16(x, y, color.Gray16{uint16(v) * 257})
			}
		}
		return []image.Image{rgba, nrgba, rgba64, nrgba64, gray, gray16}
	}

	for _, path := range []string{"mapper", "matrix"} {
		imgs := mk()
		var ref image.Image
		for i, src := range imgs {
			d := NewDitherer(blackWhite)
			if path == "mapper" {
				d.Mapper = Bayer(4, 4, 1)
			} else {
				d.Matrix = Atkinson
			}
			got := d.Dither(src)
			if i == 0 {
				ref = got
				continue
			}
			if !sameImage(ref, got) {
				t.Errorf("%s path: image type %T differs from %T", path, src, imgs[0])
			}
		}
	}
}

func TestDrawInterface(t *testing.T) {
	d := NewDitherer(blackWhite)
	d.Matrix = FloydSteinberg

	src := grayRamp(image.Point{}, 10, 10)

	// Draw the source into a larger destination at an offset, like
	// draw.Draw with a Drawer does.
	dst := image.NewRGBA(image.Rect(0, 0, 20, 20))
	d.Draw(dst, image.Rect(3, 4, 13, 14), src, image.Point{})

	// The drawn region must equal a plain dither of the source.
	want := d.Dither(grayRamp(image.Point{}, 10, 10))
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			if !sameColor(dst.At(3+x, 4+y), want.At(x, y)) {
				t.Fatalf("Draw: pixel (%d, %d) differs from Dither result", 3+x, 4+y)
			}
		}
	}

	// draw.Draw must accept the Ditherer as a draw.Drawer.
	var _ draw.Drawer = d

	// Paletted dst with the same palette works.
	pdst := image.NewPaletted(image.Rect(0, 0, 10, 10), color.Palette{color.Black, color.White})
	d.Draw(pdst, image.Rect(0, 0, 10, 10), grayRamp(image.Point{}, 10, 10), image.Point{})

	// Paletted dst with a different palette panics.
	pdst2 := image.NewPaletted(image.Rect(0, 0, 10, 10), color.Palette{
		color.RGBA{1, 2, 3, 255}, color.RGBA{4, 5, 6, 255},
	})
	assertPanics(t, "Draw with mismatched paletted dst", func() {
		d.Draw(pdst2, image.Rect(0, 0, 10, 10), src, image.Point{})
	})
}

func TestNonZeroOriginMapper(t *testing.T) {
	// The parallel mapper path must respect non-zero origins: only the
	// pixels inside the bounds are written, and the mapper receives absolute
	// coordinates.
	parent := grayRamp(image.Point{}, 20, 20)
	sub := parent.SubImage(image.Rect(5, 6, 15, 14)).(*image.RGBA)

	d := NewDitherer(blackWhite)
	d.Mapper = Bayer(4, 4, 1)
	d.Dither(sub)

	// Reference computed with absolute coordinates, on the same sub-image.
	want := refOrdered(grayRamp(image.Point{}, 20, 20).SubImage(image.Rect(5, 6, 15, 14)),
		blackWhite, refAdditiveMatrix(bayerMatrix(4, 4), 16, 1.0))
	for y := 6; y < 14; y++ {
		for x := 5; x < 15; x++ {
			if !sameColor(want.At(x, y), sub.At(x, y)) {
				t.Fatalf("pixel (%d, %d) differs from reference", x, y)
			}
		}
	}

	// Pixels outside the sub-image bounds must be untouched.
	outside := grayRamp(image.Point{}, 20, 20)
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			if x >= 5 && x < 15 && y >= 6 && y < 14 {
				continue
			}
			if !sameColor(outside.At(x, y), parent.At(x, y)) {
				t.Fatalf("pixel (%d, %d) outside the sub-image was modified", x, y)
			}
		}
	}
}
