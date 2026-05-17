package main

import fftw3 "./fftw3-odin-bindings/fftw3"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:time"
import rl "vendor:raylib"

WINDOW_HEIGHT :: 600
WINDOW_WIDTH :: 800

MinMax :: struct {
	min, max: f32,
}

normalize :: proc(x, y, x_min, x_max, y_min, y_max: f32) -> (x01, y01: f32) {
	when ODIN_DEBUG {
		assert(x_max > x_min, "Expected x_max > x_min.")
		assert(y_max >= y_min, "Expected y_max >= y_min.")
	}
	x01 = (x - x_min) / (x_max - x_min)
	x01 = clamp(x01, 0.0, 1.0)
	y_max := y_max
	if y_max == y_min {
		y_max = y_min + 1
	}
	y01 = (y - y_min) / (y_max - y_min)
	y01 = clamp(y01, 0.0, 1.0)
	return
}

map_to_rec :: proc(
	inbuf: ^[]f32,
	outbuf_line_points: [^]rl.Vector2,
	line_points_count: i32,
	outbuf_pixels: [^]rl.Vector2,
	x_min, x_max, y_min, y_max: f32,
	rec: rl.Rectangle,
) {
	n_pts := len(inbuf)

	count := make([]f32, line_points_count)
	defer delete(count)
	accu := make([]f32, line_points_count)
	defer delete(accu)

	for i in 0 ..< n_pts {
		// map i        to range [0, rec.width - 1]
		// map inbuf[i] to range [0, rec.height - 1]
		x01, y01 := normalize(f32(i), inbuf[i], x_min, x_max, y_min, y_max)
		x01 = clamp(x01, -1.0, 1.0)
		y01 = clamp(y01, -1.0, 1.0)
		x_rec_w := x01 * (rec.width - 1)
		y_rec_h := y01 * (rec.height - 1)

		// add offset so that the result lies within rec
		outbuf_pixels[i][0] = rec.x + x_rec_w
		outbuf_pixels[i][1] = rec.y + rec.height - y_rec_h // because y = 0 is top in rl

		// x_rec_w maps to this pixel column
		x_pxl := i32(math.round_f32(x_rec_w))

		// accumulate y-values in pixel columns for averaging
		accu[x_pxl] += y_rec_h
		// and count the y-values per column for averaging
		count[x_pxl] += 1
	}

	// averaging
	for x_pxl in 0 ..< line_points_count {
		if count[x_pxl] == 0 {
			count[x_pxl] += 1
		}
		// add offset so that the result lies within rec
		outbuf_line_points[x_pxl][0] = rec.x + f32(x_pxl)
		outbuf_line_points[x_pxl][1] = rec.y + rec.height - accu[x_pxl] / count[x_pxl] // because y = 0 is top in rl
	}
}

main :: proc() {
	fps: i32 = 30

	spec_x: i32 = 30
	spec_y: i32 = 10
	spec_h: i32 = WINDOW_HEIGHT / 2 - spec_y - 10
	spec_w: i32 = WINDOW_WIDTH - spec_x - 10
	spec_rec := rl.Rectangle{f32(spec_x), f32(spec_y), f32(spec_w), f32(spec_h)}
	spec_bg_color := rl.Color{40, 40, 40, 255}
	spec_line_color :: rl.Color{250, 240, 0, 255}
	spec_pixel_color :: rl.Color{250, 200, 0, 255}

	n_fft := 4096
	magnitude_max: f32 = math.F32_MIN
	magnitude_min: f32 = math.F32_MAX
	magnitude := make([]f32, n_fft)
	for &m in magnitude {
		m = rand.float32_normal(0, 1)
		if m < magnitude_min {
			magnitude_min = m
		}
		if m > magnitude_max {
			magnitude_max = m
		}
	}

	plot_line_pts_count := i32(spec_rec.width)
	plot_line_pts := make([^]rl.Vector2, plot_line_pts_count)
	plot_pixels := make([^]rl.Vector2, n_fft)

	rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Mic-Waterfall")
	defer rl.CloseWindow()

	rl.SetTargetFPS(fps)

	counter := 0
	tot: time.Tick = {0}

	for !rl.WindowShouldClose() {
		counter += 1

		tic := time.tick_now()

		magnitude_max = math.F32_MIN
		magnitude_min = math.F32_MAX
		for &m, i in magnitude {
			m = rand.float32_normal(0, 0.05) + math.sin_f32(2.0 * math.PI * f32(i) / 500)
			if m < magnitude_min {
				magnitude_min = m
			}
			if m > magnitude_max {
				magnitude_max = m
			}
		}

		map_to_rec(
			&magnitude,
			plot_line_pts,
			plot_line_pts_count,
			plot_pixels,
			0,
			f32(len(magnitude)),
			-1.1,
			1.1,
			spec_rec,
		)

		rl.BeginDrawing()
		{
			rl.ClearBackground(rl.BLACK)

			// spectrum
			rl.DrawRectangleRec(spec_rec, spec_bg_color)
			// scatter all pixels
			for i in 0 ..< n_fft {
				rl.DrawPixelV(plot_pixels[i], spec_pixel_color)
			}
			// average line plot
			rl.DrawLineStrip(plot_line_pts, plot_line_pts_count, spec_line_color)

			// horizontal splitter
			startPos: rl.Vector2 = {0, WINDOW_HEIGHT / 2}
			endPos: rl.Vector2 = {WINDOW_WIDTH, WINDOW_HEIGHT / 2}
			rl.DrawLineV(startPos, endPos, rl.WHITE)
		}
		rl.EndDrawing()

		toc := time.tick_since(tic)
		tot = time.tick_add(tot, toc)
		time_per_frame := f64(tot._nsec) / f64(counter) / 1e9
		fps := 1 / time_per_frame
		if counter % 10 == 0 {
			fmt.printf("\rFPS: %v", fps)
		}
	}
	fmt.println()
}
