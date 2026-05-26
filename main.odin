package main

import fftw3 "./fftw3-odin-bindings/fftw3"
import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:strings"
import "core:sync/chan"
import "core:thread"
import "core:time"
import ma "vendor:miniaudio"
import rl "vendor:raylib"

CHAN_CAPACITY :: 5
FFT_SIZE_R :: 4096
FFT_SIZE_C :: FFT_SIZE_R / 2 + 1
NUM_CHANNELS :: 2 // TODO: get this info from device
NUM_THREADS :: 1
SAMPLE_RATE :: 44100 // TODO: get this info from device?
WINDOW_HEIGHT :: 600
WINDOW_WIDTH :: 800

ChanTypeCallback :: []f32 // from capture callback to task thread
ChanTypeMain :: []f32 // from task thread to main

Task_Data :: struct {
	chan_callback: chan.Chan(ChanTypeCallback),
	chan_main:     chan.Chan(ChanTypeMain),
	frame_count:   u32,
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

blackman_harris_window :: proc(n: int) -> []f32 {
	a0 :: 0.4243801
	a1 :: 0.4973406
	a2 :: 0.0782793
	pi: f32 = math.PI
	win := make([]f32, n)
	n := f32(n)
	for i in 0 ..< n {
		win[int(i)] =
			a0 - a1 * math.cos_f32(2.0 * pi * i / n) + a2 * math.cos_f32(4.0 * pi * i / n)
	}
	return win
}

rb_init :: proc(rb: ^Ring_Buffer, cap: int) {
	rb.data = make([]f32, cap)
	rb.head = 0
	rb.tail = 0
	rb.count = 0
}

rb_push :: proc(rb: ^Ring_Buffer, item: f32) {
	rb.data[rb.head] = item
	rb.head += 1
	if rb.head == len(rb.data) {
		rb.head = 0
	}
	rb.count += 1
}

rb_pop :: proc(rb: ^Ring_Buffer) -> (item: f32) {
	item = rb.data[rb.tail]
	rb.tail += 1
	if rb.tail == len(rb.data) {
		rb.tail = 0
	}
	rb.count -= 1
	return
}

rb_empty_slots :: proc(rb: ^Ring_Buffer) -> int {
	return len(rb.data) - rb.count
}

rb_append :: proc(rb: ^Ring_Buffer, arr: ^[]f32) -> (success: bool) {
	success = false
	sample_count := len(arr) / NUM_CHANNELS
	if rb_empty_slots(rb) < sample_count {
		return
	}
	for i in 0 ..< sample_count {
		average: f32 = 0
		// average accross the channels
		for j in 0 ..< NUM_CHANNELS {
			average += arr[i * NUM_CHANNELS + j] / NUM_CHANNELS
		}
		rb_push(rb, average)
	}
	success = true
	return
}

proc_data :: proc(task: thread.Task) {
	fmt.printfln("[Task(%v)] start working", task.user_index)
	defer fmt.printfln("[Task(%v)] stop working", task.user_index)

	task_data := cast(^Task_Data)task.data
	chan_recv := chan.as_recv(task_data.chan_callback)
	chan_send := chan.as_send(task_data.chan_main)
	frame_count := int(task_data.frame_count)

	buf_x := make([^]f32, FFT_SIZE_R)
	buf_X := make([^]fftw3.fftwf_complex, FFT_SIZE_C)
	defer free(buf_x)
	defer free(buf_X)

	plan_forward := fftw3.fftwf_plan_dft_r2c_1d(FFT_SIZE_R, buf_x, buf_X, fftw3.Flags.ESTIMATE)
	defer fftw3.fftwf_destroy_plan(plan_forward)

	bhw := blackman_harris_window(FFT_SIZE_R)
	defer delete(bhw)

	rb := &Ring_Buffer{}
	rb_init(rb, 3 * FFT_SIZE_R)

	for {
		arr, ok := chan.recv(chan_recv)
		if !ok {
			fmt.printfln("[Task(%v)] chan_callback is closed", task.user_index)
			break
		}
		defer delete(arr)

		could_append := rb_append(rb, &arr)
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

			/* convert to dB */
			to_send := make([]f32, FFT_SIZE_C)
			for i in 0 ..< FFT_SIZE_C {
				to_send[i] = 20.0 * math.log10(fftw3.abs(buf_X[i]) + 1e-15)
			}

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

device_start :: proc(device: ^ma.device) {
	result := ma.device_start(device)
	if result != .SUCCESS {
		fmt.eprintfln("Failed to start device: %v", result)
		os.exit(1)
	}
}

device_stop :: proc(device: ^ma.device) {
	result := ma.device_stop(device)
	if result != .SUCCESS {
		fmt.eprintfln("Failed to stop device: %v", result)
		os.exit(1)
	}
}

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
			fmt.println("capture_callback: dropped frames")
		}
	}
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
	input: ^[]f32,
	output_line_points: [^]rl.Vector2,
	line_points_count: i32,
	output_pixels: [^]rl.Vector2,
	x_min, x_max, y_min, y_max: f32,
	rec: rl.Rectangle,
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
		// add offset so that the result lies within rec
		output_line_points[x_pxl][0] = rec.x + f32(x_pxl)
		output_line_points[x_pxl][1] = rec.y + rec.height - accu[x_pxl] / count[x_pxl] // because y = 0 is top in rl
	}
}

main :: proc() {
	when ODIN_DEBUG {
		context.logger = log.create_console_logger(.Debug)
	} else {
		context.logger = log.create_console_logger(.Info)
	}

	rl_fps: i32 = 30

	/* ------------------------- FFT related ------------------------- */
	magnitude_max_default: f32 = 20 * math.ceil(math.log10_f32(FFT_SIZE_R))
	magnitude_min_default: f32 = -300
	magnitude_max: f32 = magnitude_max_default
	magnitude_min: f32 = magnitude_min_default
	magnitude := make([]f32, FFT_SIZE_C)

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
	capture_config.capture.channels = NUM_CHANNELS
	capture_config.sampleRate = SAMPLE_RATE

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
	fmt.println("frame_count:", frame_count)

	capture_data: Capture_Data
	capture_data.chan_callback = chan_callback

	capture_device.pUserData = &capture_data

	// When device is started for the first time, it outputs ones followed by minus ones.
	// By briefly starting the device here, we skip this initialization phase.
	device_start(&capture_device)
	time.sleep(2_000_000)
	device_stop(&capture_device)

	/* ------------------------- thread related ------------------------- */
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, NUM_THREADS)
	defer thread.pool_destroy(&pool)

	task_data: [NUM_THREADS]Task_Data
	for i in 0 ..< NUM_THREADS {
		task_data[i] = {
			chan_callback = chan_callback,
			chan_main     = chan_main,
			frame_count   = frame_count,
		}
		thread.pool_add_task(&pool, context.allocator, proc_data, &task_data[i], i)
	}

	thread.pool_start(&pool)

	/* ------------------------- rl related ------------------------- */
	spec_x: i32 = 30
	spec_y: i32 = 10
	spec_h: i32 = WINDOW_HEIGHT / 2 - spec_y - 10
	spec_w: i32 = WINDOW_WIDTH - spec_x - 10
	spec_rec := rl.Rectangle{f32(spec_x), f32(spec_y), f32(spec_w), f32(spec_h)}
	spec_bg_color := rl.Color{40, 40, 40, 255}
	spec_line_color :: rl.Color{250, 240, 0, 255}
	spec_pixel_color :: rl.Color{250, 200, 0, 255}

	plot_line_pts_count := i32(spec_rec.width)
	plot_line_pts := make([^]rl.Vector2, plot_line_pts_count)
	plot_pixels := make([^]rl.Vector2, FFT_SIZE_C)

	rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Mic-Waterfall")
	defer rl.CloseWindow()

	rl.SetTargetFPS(rl_fps)

	/* ------------------------- main loop ------------------------- */
	auto_scale_y := false
	capturing := false
	fps: f64 = 0.0
	fps_str := strings.clone_to_cstring(fmt.aprintf("FPS: %.2f", fps))
	rd_avail: f32 = 0.0
	frames_count := 0
	frames_start_time := time.tick_now()
	for !rl.WindowShouldClose() {
		frames_count += 1

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
		}
		if rl.IsKeyPressed(.R) {
			magnitude_max = magnitude_max_default
			magnitude_min = magnitude_min_default
		}

		arr, ok := chan.try_recv(chan_recv)
		defer delete(arr)
		if ok {
			magnitude = arr

			if auto_scale_y {
				magnitude_max = math.F32_MIN
				magnitude_min = math.F32_MAX
				for &m in magnitude {
					if m < magnitude_min {
						magnitude_min = m
					}
					if m > magnitude_max {
						magnitude_max = m
					}
				}
				auto_scale_y = false
			}

			map_to_rec(
				&magnitude,
				plot_line_pts,
				plot_line_pts_count,
				plot_pixels,
				0,
				f32(len(magnitude) / 2 + 1),
				magnitude_min,
				magnitude_max,
				spec_rec,
			)
		}

		rl.BeginDrawing()
		{
			rl.ClearBackground(rl.BLACK)

			/* spectrum */
			rl.DrawRectangleRec(spec_rec, spec_bg_color)
			// scatter all pixels
			for i in 0 ..< FFT_SIZE_C {
				rl.DrawPixelV(plot_pixels[i], spec_pixel_color)
			}
			// average line plot
			rl.DrawLineStrip(plot_line_pts, plot_line_pts_count, spec_line_color)

			/* horizontal splitter */
			startPos: rl.Vector2 = {0, WINDOW_HEIGHT / 2}
			endPos: rl.Vector2 = {WINDOW_WIDTH, WINDOW_HEIGHT / 2}
			rl.DrawLineV(startPos, endPos, rl.WHITE)

			/* frames per second */
			if frames_count % 10 == 0 {
				frames_total_time := f64(time.tick_since(frames_start_time)) / 1.0e9
				time_per_frame := frames_total_time / f64(frames_count)
				fps = 1 / time_per_frame
				fps_str = strings.clone_to_cstring(fmt.aprintf("FPS: %.2f", fps))
			}
			rl.DrawText(fps_str, 20, WINDOW_HEIGHT / 2 + 20, 20, rl.WHITE)
		}
		rl.EndDrawing()
	}
	chan.close(chan_callback)
	chan.close(chan_main)
	thread.pool_finish(&pool)
}
