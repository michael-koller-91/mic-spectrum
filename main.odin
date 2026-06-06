package main

/* TODO:
  * vertical grid seems to be off by one pixel
  * single spectrum pixels can be one pixel below lowest horizontal line
*/

/*NOTE:
  * anything font-related needs to be called after rl.InitWindow
  * anything texture-related needs to be called after rl.InitWindow
  * averaging the channels does not seem to be a good idea (at least with my mic), so 1 channel for now
    * some kind of phase alignment prior to averaging would be necessary
*/

MEMTRACK :: #config(MEMTRACK, false)

import fftw3 "./fftw3-odin-bindings/fftw3"
import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math"
import "core:mem"
import "core:os"
import "core:strings"
import "core:sync/chan"
import "core:thread"
import "core:time"
import ma "vendor:miniaudio"
import rl "vendor:raylib"

CHAN_CAPACITY :: 5 // capacity of the channels between the threads
FFT_SIZE_R :: 4096
FFT_SIZE_C :: FFT_SIZE_R / 2 + 1
WINDOW_HEIGHT :: 800
WINDOW_WIDTH :: 1000

ChanTypeCallback :: []f32 // from capture callback to DSP thread
ChanTypeMain :: []f32 // from DSP thread to main

DSP_Data :: struct {
	chan_callback: chan.Chan(ChanTypeCallback),
	chan_main:     chan.Chan(ChanTypeMain),
	num_channels:  int,
}

Capture_Data :: struct {
	chan_callback: chan.Chan(ChanTypeCallback),
}

Ring_Buffer :: struct {
	data:  []f32,
	head:  int, // index of slot where to write next
	tail:  int, // index of slot where to read next
	count: int, // number of items in the buffer
}

/*
A Blackman-Harris window of length `length`.
*/
blackman_harris_window :: proc(length: int) -> []f32 {
	a0 :: 0.4243801
	a1 :: 0.4973406
	a2 :: 0.0782793
	pi: f32 = math.PI
	win := make([]f32, length)
	n := f32(length)
	for i in 0 ..< n {
		theta := 2.0 * pi * i / n
		win[int(i)] = a0 - a1 * math.cos_f32(theta) + a2 * math.cos_f32(2.0 * theta)
	}
	return win
}

/*
Something resembling the colormap plasma.

dark purple   : (0.05, 0.00, 0.50)
medium purple : (0.25, 0.00, 0.75)
bright pink   : (0.75, 0.00, 0.25)
yellow        : (1.00, 1.00, 0.00)
*/
color_map :: proc(x: f32) -> (rgba: rl.Color) {
	x := x
	rgba.a = 255
	r, g, b: f32
	if x < 0.20 {
		x = x / 0.20
		r = 0.05 + 0.2 * x
		g = 0
		b = 0.5 + 0.25 * x
	} else if x < 0.66 {
		x = (x - 0.20) / 0.46
		r = 0.25 + 0.5 * x
		g = 0
		b = 0.75 - 0.5 * x
	} else {
		x = (x - 0.66) / 0.34
		r = 0.75 + 0.25 * x
		g = x
		b = 0.25 - 0.25 * x
	}
	rgba.r = u8(math.round(255 * r))
	rgba.g = u8(math.round(255 * g))
	rgba.b = u8(math.round(255 * b))
	return
}

/*
Initialize the ring buffer `rb` with capacity `cap`.
*/
rb_make :: proc(rb: ^Ring_Buffer, cap: int) {
	rb.data = make([]f32, cap)
	rb.head = 0
	rb.tail = 0
	rb.count = 0
}

/*
Delete the ring buffer.
*/
rb_delete :: proc(rb: ^Ring_Buffer) {
	delete(rb.data)
}

/*
Push a value to the ring buffer (push back).
*/
rb_push :: proc(rb: ^Ring_Buffer, item: f32) {
	rb.data[rb.head] = item
	rb.head += 1
	if rb.head == len(rb.data) {
		rb.head = 0
	}
	rb.count += 1
}

/*
Pop a value from the ring buffer (pop front).
*/
rb_pop :: proc(rb: ^Ring_Buffer) -> (item: f32) {
	item = rb.data[rb.tail]
	rb.tail += 1
	if rb.tail == len(rb.data) {
		rb.tail = 0
	}
	rb.count -= 1
	return
}

/*
The number of slots still empty in the ring buffer.
*/
rb_empty_slots :: proc(rb: ^Ring_Buffer) -> int {
	return len(rb.data) - rb.count
}

/*
Append a whole array to the ring buffer.
*/
rb_append :: proc(rb: ^Ring_Buffer, arr: ^[]f32, num_channels: int) -> (success: bool) {
	success = false
	sample_count := len(arr) / num_channels
	if rb_empty_slots(rb) < sample_count {
		return
	}
	for i in 0 ..< sample_count {
		average: f32 = 0
		// average accross the channels
		for j in 0 ..< num_channels {
			average += arr[i * num_channels + j] / f32(num_channels)
		}
		rb_push(rb, average)
	}
	success = true
	return
}

/*
The DSP thread processes the microphone samples to produce spectrum values.
*/
dsp :: proc(task: thread.Task) {
	fmt.printfln("[Task(%v)] start working", task.user_index)
	defer fmt.printfln("[Task(%v)] stop working", task.user_index)

	when MEMTRACK {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.temp_allocator)
		context.temp_allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf(
					"[Task(%v)] === %v context.temp_allocator allocations not freed: ===\n",
					task.user_index,
					len(track.allocation_map),
				)
				for _, entry in track.allocation_map {
					fmt.eprintf(
						"(%v) - %v bytes @ %v\n",
						task.user_index,
						entry.size,
						entry.location,
					)
				}
			} else {
				fmt.printfln(
					"[Task(%v)] === context.temp_allocator tracking was active (no missed frees) ===",
					task.user_index,
				)
			}
			mem.tracking_allocator_destroy(&track)
		}
	}

	dsp_data := cast(^DSP_Data)task.data
	chan_recv := chan.as_recv(dsp_data.chan_callback) // receive from capture callback
	chan_send := chan.as_send(dsp_data.chan_main) // send to main thread
	num_channels := int(dsp_data.num_channels)

	buf_x := make([^]f32, FFT_SIZE_R)
	buf_X := make([^]fftw3.fftwf_complex, FFT_SIZE_C)
	defer free(buf_x)
	defer free(buf_X)

	plan_forward := fftw3.fftwf_plan_dft_r2c_1d(FFT_SIZE_R, buf_x, buf_X, fftw3.Flags.ESTIMATE)
	defer fftw3.fftwf_destroy_plan(plan_forward)

	bhw := blackman_harris_window(FFT_SIZE_R)
	defer delete(bhw)

	rb := &Ring_Buffer{}
	rb_make(rb, 3 * FFT_SIZE_R)
	defer rb_delete(rb)

	for {
		arr, ok := chan.recv(chan_recv)
		if !ok {
			fmt.printfln("[Task(%v)] chan_callback is closed", task.user_index)
			break
		}

		could_append := rb_append(rb, &arr, num_channels)
		if !could_append {
			fmt.printfln("[Task(%v)] ring-buffer too full, dropped data", task.user_index)
		}

		if rb.count > FFT_SIZE_R {
			/* read data with 50% overlap */
			// read and pop half of the array
			for i in 0 ..< FFT_SIZE_R / 2 {
				buf_x[i] = rb_pop(rb)
			}
			// read but don't pop the other half of the array
			count_saved := rb.count
			tail_saved := rb.tail
			for i in FFT_SIZE_R / 2 ..< FFT_SIZE_R {
				buf_x[i] = rb_pop(rb)
			}
			// restore the previous state, i.e., undo pop
			rb.count = count_saved
			rb.tail = tail_saved

			/* windowing */
			for i in 0 ..< FFT_SIZE_R {
				buf_x[i] *= bhw[i]
			}

			/* FFT */
			fftw3.fftwf_execute_dft_r2c(plan_forward, buf_x, buf_X)

			/* convert to dBFS */
			to_send := make([]f32, FFT_SIZE_C)
			for i in 0 ..< FFT_SIZE_C {
				to_send[i] = 20.0 * math.log10(fftw3.abs(buf_X[i]) + 1e-15)
			}

			/* send to main thread */
			if chan.len(chan_send) < CHAN_CAPACITY - 2 {
				success := chan.send(chan_send, to_send)
				if !success {
					fmt.eprintfln(
						"[Task(%v)] ERROR: Failed to send to chan_send.",
						task.user_index,
					)
					os.exit(1)
				}
			} else {
				fmt.printfln("[Task(%v)] dropped data", task.user_index)
			}
		}
	}
}

/*
Start capturing microphone samples.
*/
device_start :: proc(device: ^ma.device) {
	result := ma.device_start(device)
	if result != .SUCCESS {
		fmt.eprintfln("Failed to start device: %v", result)
		os.exit(1)
	}
}

/*
Stop capturing microphone samples.
*/
device_stop :: proc(device: ^ma.device) {
	result := ma.device_stop(device)
	if result != .SUCCESS {
		fmt.eprintfln("Failed to stop device: %v", result)
		os.exit(1)
	}
}

/*
Callback to capture microphone samples. Frames of samples are sent to the worker task.
*/
capture_callback :: proc "c" (device: ^ma.device, output, input: rawptr, frame_count: u32) {
	context = runtime.default_context()

	data_ptr: rawptr
	capture_data := cast(^Capture_Data)device.pUserData
	chan_send := chan.as_send(capture_data.chan_callback)

	if frame_count > 0 {
		if chan.len(chan_send) < CHAN_CAPACITY - 1 {
			arr := make([]f32, frame_count)
			mem.copy(raw_data(arr), input, int(frame_count) * size_of(f32))
			success := chan.send(chan_send, arr)
			if !success {
				fmt.eprintln("ERROR: capture_callback: Failed to send to chan_send.")
				os.exit(1)
			}
		} else {
			fmt.printfln("capture_callback: dropped %v frames", frame_count)
		}
	}
}

/*
Convert x from range [x_min, x_max] to range [0, 1].
Convert y from range [y_min, y_max] to range [0, 1].
*/
normalize :: proc(x, y, x_min, x_max, y_min, y_max: f32) -> (x01, y01: f32) {
	when ODIN_DEBUG {
		// this is super slow, so only in debug
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

/*
Map the FFT values to pixels within the rectangle defined by `rec`.
*/
map_to_rec :: proc(
	input: ^[]f32,
	line_points_count: i32,
	x_min, x_max, y_min, y_max: f32,
	rec: rl.Rectangle,
	output_line_points: [^]rl.Vector2,
	output_pixels: [^]rl.Vector2,
	output_raw_avg: ^[]f32,
) {
	n_pts := len(input)

	count := make([]f32, line_points_count)
	defer delete(count)
	accu := make([]f32, line_points_count)
	defer delete(accu)

	for i in 0 ..< n_pts {
		// map i        to range [0, rec.width - 1]
		// map input[i] to range [0, rec.height - 1]
		x01, y01 := normalize(f32(i), input[i], x_min, x_max, y_min, y_max)
		x01 = clamp(x01, 0, 1.0)
		y01 = clamp(y01, 0, 1.0)
		x_rec_w := x01 * (rec.width - 1)
		y_rec_h := y01 * (rec.height - 1)

		// add offset so that the result lies within rec
		output_pixels[i][0] = rec.x + x_rec_w
		output_pixels[i][1] = rec.y + rec.height - y_rec_h // because y = 0 is top in rl

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
		output_raw_avg[x_pxl] = accu[x_pxl] / count[x_pxl]
		// add offset so that the result lies within rec
		output_line_points[x_pxl][0] = rec.x + f32(x_pxl)
		output_line_points[x_pxl][1] = rec.y + rec.height - output_raw_avg[x_pxl] // because y = 0 is top in rl
	}
}

main :: proc() {
	when ODIN_DEBUG {
		context.logger = log.create_console_logger(.Debug)
	} else {
		context.logger = log.create_console_logger(.Info)
	}

	when MEMTRACK {
		track1: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track1, context.allocator)
		context.allocator = mem.tracking_allocator(&track1)

		defer {
			if len(track1.allocation_map) > 0 {
				fmt.eprintf(
					"=== %v context.allocator allocations not freed: ===\n",
					len(track1.allocation_map),
				)
				for _, entry in track1.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			} else {
				fmt.printfln("=== context.allocator tracking was active (no missed frees) ===")
			}
			mem.tracking_allocator_destroy(&track1)
		}

		track2: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track2, context.temp_allocator)
		context.temp_allocator = mem.tracking_allocator(&track2)

		defer {
			if len(track2.allocation_map) > 0 {
				fmt.eprintf(
					"=== %v context.temp_allocator allocations not freed: ===\n",
					len(track2.allocation_map),
				)
				for _, entry in track2.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			} else {
				fmt.printfln(
					"=== context.temp_allocator tracking was active (no missed frees) ===",
				)
			}
			mem.tracking_allocator_destroy(&track2)
		}
	}

	/* ------------------------- FFT related ------------------------- */
	mag_max_default: f32 = 20 * math.ceil(math.log10_f32(FFT_SIZE_R))
	mag_min_default: f32 = -100
	mag_max: f32 = mag_max_default
	mag_min: f32 = mag_min_default

	/* ------------------------- channel related ------------------------- */
	chan_callback, err_callback := chan.create(
		chan.Chan(ChanTypeCallback),
		CHAN_CAPACITY,
		context.allocator,
	)
	assert(err_callback == .None)
	defer chan.destroy(chan_callback)

	chan_main, err_main := chan.create(chan.Chan(ChanTypeMain), CHAN_CAPACITY, context.allocator)
	assert(err_main == .None)
	defer chan.destroy(chan_main)
	chan_recv := chan.as_recv(chan_main)

	/* ------------------------- ma related ------------------------- */
	result: ma.result

	capture_config := ma.device_config_init(.capture)
	capture_config.dataCallback = capture_callback
	capture_config.capture.format = .f32
	capture_config.capture.channels = 1 // see NOTE
	capture_config.sampleRate = 0 // let ma choose the native sample rate

	capture_device: ma.device
	result = ma.device_init(nil, &capture_config, &capture_device)
	if result != .SUCCESS {
		fmt.eprintln("Failed to initialize capture_device:", result)
		os.exit(1)
	}
	log.debug("Initialized capture_device")
	defer ma.device_uninit(&capture_device)
	defer log.debug("Uninitialized capture_device")

	frame_count := capture_device.capture.internalPeriodSizeInFrames
	num_channels := int(capture_device.capture.internalChannels)
	sample_rate := capture_device.capture.internalSampleRate
	fmt.println("frame_count:", frame_count)
	fmt.println("num_channels:", num_channels)
	fmt.println("sample rate:", sample_rate)

	capture_data: Capture_Data
	capture_data.chan_callback = chan_callback
	capture_device.pUserData = &capture_data

	/* ------------------------- thread related ------------------------- */
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, 1)
	defer thread.pool_destroy(&pool)
	dsp_data := DSP_Data {
		chan_callback = chan_callback,
		chan_main     = chan_main,
		num_channels  = num_channels,
	}
	thread.pool_add_task(&pool, context.allocator, dsp, &dsp_data, 0)

	thread.pool_start(&pool)

	/* ------------------------- rl related ------------------------- */
	rl_fps: i32 = 30

	font_spacing: f32 = 0
	font_cstr := strings.clone_to_cstring("DMMono-Regular.ttf")
	defer delete(font_cstr)

	// heights of the three displayed sections
	h1: i32 = 40
	h2: i32 = (WINDOW_HEIGHT - h1) / 2
	h3: i32 = WINDOW_HEIGHT - h2 - h1

	// colors
	grid_color :: rl.Color{100, 100, 100, 180}
	plot_bg_color := rl.Color{40, 40, 40, 255}
	spec_line_color :: rl.Color{250, 240, 0, 255}
	spec_pixel_color :: rl.Color{250, 200, 0, 255}
	text_color :: rl.WHITE
	tick_color :: text_color

	tick_len: f32 = 10
	yaxis_unit_x: i32 = 30

	// spectrum
	spec_x: i32 = 100 // offset from left to where spectrum starts
	spec_y: i32 = h1 + 20 // offset from top to where spectrum starts
	spec_h: i32 = h2 - 40
	spec_w: i32 = WINDOW_WIDTH - spec_x - 80
	spec_rec := rl.Rectangle{f32(spec_x), f32(spec_y), f32(spec_w), f32(spec_h)}

	spec_yaxis_unit_cstr := strings.clone_to_cstring("[dBFS]")
	defer delete(spec_yaxis_unit_cstr)
	spec_yaxis_unit_y: i32 = spec_y + spec_h / 2

	plot_line_pts_count := i32(spec_rec.width)
	plot_line_pts := make([^]rl.Vector2, plot_line_pts_count)
	plot_line_avg := make([]f32, plot_line_pts_count)
	plot_pixels := make([^]rl.Vector2, FFT_SIZE_C)
	defer {
		free(plot_line_pts)
		delete(plot_line_avg)
		free(plot_pixels)
	}

	rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Microphone Spectrum")
	defer rl.CloseWindow()

	font := rl.LoadFontEx(font_cstr, 21, nil, 0)
	defer rl.UnloadFont(font)
	if !rl.IsFontValid(font) {
		font = rl.GetFontDefault()
		fmt.println("INFO: Using default font.")
	}

	xtick_num: i32 = 10
	xtick_text := make([]cstring, xtick_num)
	xtick_text_len := make([]rl.Vector2, xtick_num)
	for x in 0 ..< xtick_num {
		frequency := f32(x) * f32(sample_rate) / 2 / f32(xtick_num - 1)
		t := fmt.aprintf("%.0f", frequency)
		xtick_text[x] = strings.clone_to_cstring(t)
		delete(t)
		xtick_text_len[x] = rl.MeasureTextEx(font, xtick_text[x], f32(font.baseSize), font_spacing)
	}
	defer delete(xtick_text)
	defer delete(xtick_text_len)
	defer for x in 0 ..< xtick_num {
		delete(xtick_text[x])
	}

	xaxis_unit_cstr := strings.clone_to_cstring("[Hz]")
	defer delete(xaxis_unit_cstr)
	xaxis_unit_len := rl.MeasureTextEx(font, xaxis_unit_cstr, f32(font.baseSize), font_spacing)

	spec_ytick_num: i32 = 5
	spec_ytick_text := make([]cstring, spec_ytick_num)
	spec_ytick_text_len := make([]rl.Vector2, spec_ytick_num)
	for y in 0 ..< spec_ytick_num {
		yy := spec_ytick_num - 1 - y
		db := f32(y) * (mag_max_default - mag_min_default) + mag_min_default
		t := fmt.aprintf("%.0f", db)
		spec_ytick_text[yy] = strings.clone_to_cstring(t)
		delete(t)
		spec_ytick_text_len[yy] = rl.MeasureTextEx(
			font,
			spec_ytick_text[yy],
			f32(font.baseSize),
			font_spacing,
		)
	}
	defer delete(spec_ytick_text)
	defer delete(spec_ytick_text_len)
	defer for y in 0 ..< spec_ytick_num {
		delete(spec_ytick_text[y])
	}

	wf_x := spec_x
	wf_y: i32 = h1 + h2 + 20 + i32(tick_len / 2) + font.baseSize

	// texture for waterfall
	rt_wf := rl.LoadRenderTexture(spec_w, h3 - (wf_y - h1 - h2) - 20)
	rt_wf_tmp := rl.LoadRenderTexture(rt_wf.texture.width, rt_wf.texture.height)
	defer rl.UnloadRenderTexture(rt_wf)
	defer rl.UnloadRenderTexture(rt_wf_tmp)

	wf_yaxis_unit_cstr := strings.clone_to_cstring("[s]")
	defer delete(wf_yaxis_unit_cstr)
	wf_yaxis_unit_y: i32 = wf_y + rt_wf.texture.height / 2

	cm_w: i32 = 40
	cm_x: i32 = spec_x + spec_w + (WINDOW_WIDTH - spec_x - spec_w - cm_w) / 2
	cm_y: i32 = spec_y

	time_per_slice: f32 = f32(num_channels) * f32(FFT_SIZE_R) / 2 / f32(sample_rate) // time between two STFT slices
	time_tot_wf: f32 = f32(rt_wf.texture.height) * time_per_slice // how much time the whole waterfall displays
	time_tot: f32 = math.floor(time_tot_wf)
	wf_ytick_num := i32(math.floor(time_tot_wf / 2) + 1) // one label every 2 seconds + second 0

	wf_ytick_text := make([]cstring, wf_ytick_num)
	wf_ytick_meas := make([]rl.Vector2, wf_ytick_num)
	for y in 0 ..< wf_ytick_num {
		seconds := y * 2
		t := fmt.aprintf("%d", seconds)
		wf_ytick_text[y] = strings.clone_to_cstring(t)
		delete(t)
		wf_ytick_meas[y] = rl.MeasureTextEx(
			font,
			wf_ytick_text[y],
			f32(font.baseSize),
			font_spacing,
		)
	}
	defer delete(wf_ytick_text)
	defer delete(wf_ytick_meas)
	defer for y in 0 ..< wf_ytick_num {
		delete(wf_ytick_text[y])
	}

	keybindings_cstr := strings.clone_to_cstring(
		"[SPACE]: capturing is active while held | [a] auto-scale y-axis | [r] reset y-axis | [ESC] exit",
	)
	defer delete(keybindings_cstr)

	// texture for colormap
	rt_cm := rl.LoadRenderTexture(cm_w, spec_h)
	defer rl.UnloadRenderTexture(rt_cm)
	// visualize colormap
	rl.BeginTextureMode(rt_cm)
	{
		rl.ClearBackground(rl.BLANK)
		for y in 0 ..< rt_cm.texture.height {
			for x in 0 ..< rt_cm.texture.width {
				rl.DrawPixel(i32(x), i32(y), color_map(f32(y) / f32(rt_cm.texture.height)))
			}
		}
	}
	rl.EndTextureMode()

	rl.SetTargetFPS(rl_fps)

	/* ------------------------- main loop ------------------------- */
	auto_scale_y := false
	capturing := false
	update_ylabels := false

	fps: f64 = 0.0
	fps_str := fmt.aprintf("FPS: %.1f", fps)
	defer delete(fps_str)
	fps_cstr := strings.clone_to_cstring(fps_str)
	defer delete(fps_cstr)

	rd_avail: f32 = 0.0
	frames_count := 0
	frames_start_time := time.tick_now()
	for !rl.WindowShouldClose() {
		if rl.IsKeyPressed(.SPACE) {
			rd_avail = 0
			capturing = true
			device_start(&capture_device)
		}
		if rl.IsKeyReleased(.SPACE) {
			capturing = false
			device_stop(&capture_device)
		}
		if rl.IsKeyPressed(.A) {
			auto_scale_y = true
			update_ylabels = true
		}
		if rl.IsKeyPressed(.R) {
			mag_max = mag_max_default
			mag_min = mag_min_default
			update_ylabels = true
		}

		mag, ok := chan.try_recv(chan_recv)
		defer delete(mag)
		if ok {
			if auto_scale_y {
				mag_max = math.F32_MIN
				mag_min = math.F32_MAX
				for &m in mag {
					if m < mag_min {
						mag_min = m
					}
					if m > mag_max {
						mag_max = m
					}
				}
				mag_max *= 0.9
				mag_min *= 1.1
				auto_scale_y = false
			}

			if update_ylabels {
				// update spectrum y-axis labels
				for y in 0 ..< spec_ytick_num {
					yy := spec_ytick_num - 1 - y
					db := f32(y) * (mag_max - mag_min) + mag_min
					t := fmt.aprintf("%.0f", db)
					delete(spec_ytick_text[yy])
					spec_ytick_text[yy] = strings.clone_to_cstring(t)
					delete(t)
					spec_ytick_text_len[yy] = rl.MeasureTextEx(
						font,
						spec_ytick_text[yy],
						f32(font.baseSize),
						font_spacing,
					)

				}
				update_ylabels = false
			}

			map_to_rec(
				&mag,
				plot_line_pts_count,
				0,
				f32(len(mag)),
				mag_min,
				mag_max,
				spec_rec,
				plot_line_pts,
				plot_pixels,
				&plot_line_avg,
			)

			// draw texture one pixel down
			rl.BeginTextureMode(rt_wf_tmp)
			{
				rl.ClearBackground(rl.BLANK)
				rl.DrawTexture(rt_wf.texture, 0, 1, rl.WHITE)
			}
			rl.EndTextureMode()

			rl.BeginTextureMode(rt_wf)
			{
				// draw texture
				rl.DrawTexture(rt_wf_tmp.texture, 0, 0, rl.WHITE)
				// draw a new line at the top
				for x in 0 ..< rt_wf.texture.width {
					x01 := plot_line_avg[x] / (spec_rec.height - 1)
					rl.DrawPixel(x, rt_wf.texture.height - 1, color_map(x01))
				}
			}
			rl.EndTextureMode()
		}

		// update FPS
		frames_count += 1
		if frames_count % 10 == 0 {
			frames_total_time := f64(time.tick_since(frames_start_time)) / 1.0e9
			time_per_frame := frames_total_time / f64(frames_count)
			fps = 1 / time_per_frame
			delete(fps_str)
			fps_str = fmt.aprintf("FPS: %.1f", fps)
			delete(fps_cstr)
			fps_cstr = strings.clone_to_cstring(fps_str)
		}

		rl.BeginDrawing()
		{
			rl.ClearBackground(rl.BLACK)

			/* keybindings */
			rl.DrawTextEx(
				font,
				keybindings_cstr,
				rl.Vector2{10, 10},
				f32(font.baseSize),
				font_spacing,
				text_color,
			)

			/* FPS */
			rl.DrawTextEx(
				font,
				fps_cstr,
				rl.Vector2{f32(WINDOW_WIDTH - 100), f32(10)},
				f32(font.baseSize),
				font_spacing,
				rl.GRAY,
			)

			/* upper horizontal splitter */
			rl.DrawLineV(rl.Vector2{0, f32(h1)}, rl.Vector2{WINDOW_WIDTH, f32(h1)}, rl.WHITE)

			/* spectrum rectangle */
			rl.DrawRectangleRec(spec_rec, plot_bg_color)

			/* [dBFS] */
			dbfs_orig := rl.MeasureTextEx(
				font,
				spec_yaxis_unit_cstr,
				f32(font.baseSize),
				font_spacing,
			)
			dbfs_orig.x /= 2
			dbfs_orig.y /= 2
			rl.DrawTextPro(
				font,
				spec_yaxis_unit_cstr,
				rl.Vector2{f32(yaxis_unit_x), f32(spec_yaxis_unit_y)},
				dbfs_orig,
				-90,
				f32(font.baseSize),
				font_spacing,
				text_color,
			)

			/* spectrum y-axis labels */
			ytick_left: rl.Vector2 = {f32(spec_x) - tick_len / 2, 0}
			ytick_right: rl.Vector2 = {f32(spec_x) + tick_len / 2, 0}
			for y in 0 ..< spec_ytick_num {
				ytick_left.y = f32(spec_y) + f32(y) * f32(spec_h) / f32(spec_ytick_num - 1)
				ytick_right.y = f32(spec_y) + f32(y) * f32(spec_h) / f32(spec_ytick_num - 1)
				// horizontal grid
				rl.DrawLine(
					spec_x,
					i32(ytick_left.y),
					spec_x + spec_w,
					i32(ytick_left.y),
					grid_color,
				)
				// text
				rl.DrawTextEx(
					font,
					spec_ytick_text[y],
					rl.Vector2 {
						ytick_left.x - spec_ytick_text_len[y].x - f32(font.baseSize) / 2,
						ytick_left.y - spec_ytick_text_len[y].y / 2,
					},
					f32(font.baseSize),
					font_spacing,
					tick_color,
				)
				// tick
				rl.DrawLineV(ytick_left, ytick_right, tick_color)
			}

			/* spectrum */
			// scatter all pixels
			for i in 0 ..< FFT_SIZE_C {
				rl.DrawPixelV(plot_pixels[i], spec_pixel_color)
			}
			// average line plot
			rl.DrawLineStrip(plot_line_pts, plot_line_pts_count, spec_line_color)

			/* horizontal axis text, grid, tick */
			xtick_lower: rl.Vector2 = {0, f32(h1 + h2) + tick_len / 2}
			xtick_upper: rl.Vector2 = {0, f32(h1 + h2) - tick_len / 2}
			for x in 0 ..< xtick_num {
				xtick_lower.x = f32(spec_x) + f32(x) * f32(spec_w) / f32(xtick_num - 1)
				xtick_upper.x = f32(spec_x) + f32(x) * f32(spec_w) / f32(xtick_num - 1)
				// text
				rl.DrawTextEx(
					font,
					xtick_text[x],
					rl.Vector2 {
						xtick_lower.x - xtick_text_len[x].x / 2,
						xtick_lower.y + xtick_text_len[x].y / 2,
					},
					f32(font.baseSize),
					font_spacing,
					tick_color,
				)
				// vertical grid
				rl.DrawLine(
					i32(xtick_lower.x),
					spec_y,
					i32(xtick_lower.x),
					spec_y + spec_h,
					grid_color,
				)
				// tick
				rl.DrawLineV(xtick_lower, xtick_upper, tick_color)
			}
			// [Hz]
			rl.DrawTextEx(
				font,
				xaxis_unit_cstr,
				rl.Vector2 {
					f32(WINDOW_WIDTH) - 1.4 * xaxis_unit_len.x,
					xtick_lower.y + xaxis_unit_len.y / 2,
				},
				f32(font.baseSize),
				font_spacing,
				text_color,
			)

			/* lower horizontal splitter */
			rl.DrawLineV(
				rl.Vector2{0, f32(h1 + h2)},
				rl.Vector2{WINDOW_WIDTH, f32(h1 + h2)},
				rl.WHITE,
			)

			/* box behind waterfall */
			rl.DrawRectangle(wf_x, wf_y, rt_wf.texture.width, rt_wf.texture.height, plot_bg_color)

			/* waterfall */
			rl.DrawTexture(rt_wf.texture, wf_x, wf_y, rl.WHITE)

			/* waterfall times */
			ytick_left = {f32(wf_x) - tick_len / 2, 0}
			ytick_right = {f32(wf_x) + tick_len / 2, 0}
			y_2_seconds := math.round(f32(rt_wf.texture.height) * 2.0 / time_tot_wf)
			for y in 0 ..< wf_ytick_num {
				ytick_left.y = f32(wf_y) + f32(y) * y_2_seconds
				ytick_right.y = f32(wf_y) + f32(y) * y_2_seconds
				// grid
				rl.DrawLine(
					wf_x,
					i32(ytick_left.y),
					wf_x + rt_wf.texture.width,
					i32(ytick_left.y),
					grid_color,
				)
				// text
				rl.DrawTextEx(
					font,
					wf_ytick_text[y],
					rl.Vector2 {
						ytick_left.x - wf_ytick_meas[y].x - f32(font.baseSize) / 2,
						ytick_left.y - wf_ytick_meas[y].y / 2,
					},
					f32(font.baseSize),
					font_spacing,
					tick_color,
				)
				// tick
				rl.DrawLineV(ytick_left, ytick_right, tick_color)
			}

			/* waterfall vertical grid */
			xpos: f32
			for x in 0 ..< xtick_num {
				xpos = f32(wf_x) + f32(x) * f32(rt_wf.texture.width) / f32(xtick_num - 1)
				rl.DrawLine(i32(xpos), wf_y, i32(xpos), wf_y + rt_wf.texture.height, grid_color)
			}

			/* [s] */
			s_pos := rl.Vector2{f32(yaxis_unit_x), f32(wf_yaxis_unit_y)}
			s_orig := rl.MeasureTextEx(font, wf_yaxis_unit_cstr, f32(font.baseSize), font_spacing)
			s_orig.x /= 2
			s_orig.y /= 2
			rl.DrawTextPro(
				font,
				wf_yaxis_unit_cstr,
				s_pos,
				s_orig,
				-90,
				f32(font.baseSize),
				font_spacing,
				text_color,
			)

			/* colormap */
			rl.DrawTexture(rt_cm.texture, cm_x, cm_y, rl.WHITE)
		}
		rl.EndDrawing()
	}
	chan.close(chan_callback)
	chan.close(chan_main)
	thread.pool_finish(&pool)
}
