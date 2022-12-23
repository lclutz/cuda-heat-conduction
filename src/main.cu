#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>

#include <X11/Xlib.h>

#include <GL/gl.h>
#include <GL/glx.h>

#define IMGUI_DISABLE_SSE
#include "imgui.h"
#include "imgui_impl_xlib.h"
#include "imgui_impl_opengl2.h"

#include "imgui.cpp"
#include "imgui_draw.cpp"
#include "imgui_tables.cpp"
#include "imgui_widgets.cpp"
#include "imgui_impl_xlib.cpp"
#include "imgui_impl_opengl2.cpp"

#include "defer.h"
#include "state.h"
#include "logging.h"

// Note: We are controlling the frame rate of the application to make it behave
//       nicely if run locally. Because we mostly run it over the network the
//       frame rates we are actually getting are much lower than this.
constexpr float TARGET_FPS        = 60.0f;
constexpr auto  TARGET_FRAME_TIME = std::chrono::duration<float>(1.0f/TARGET_FPS);

constexpr int   WINDOW_WIDTH  = 800;
constexpr int   WINDOW_HEIGHT = 600;

__global__ void
heat_conduction_kernel(
    int width, int height, float alphaTimesDt,
    float* temp_in, float* temp_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = idx % width;
    int row = idx / width;

	if (0 < row && row < height-1 && 0 < col && col < width-1) {
		int left  = idx - 1;
		int right = idx + 1;
		int up    = idx - width;
		int down  = idx + width;

		// evaluate derivatives
		float d2tdx2 = temp_in[left] - 2.0f * temp_in[idx] + temp_in[right];
		float d2tdy2 = temp_in[up]   - 2.0f * temp_in[idx] + temp_in[down];

        if (d2tdx2 < 1e-5) { d2tdx2 = 0.0f; }
        if (d2tdy2 < 1e-5) { d2tdy2 = 0.0f; }

		// update temperature
        temp_out[idx] = temp_in[idx] + alphaTimesDt * (d2tdx2 + d2tdy2);
	}
}

__global__ void
color_kernel(
    int width, int height, float lighter_temp,
    float *temps, uint32_t *pixels_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width * height) {
        float norm_temp = temps[idx]/lighter_temp;

        // Pixel format: 0xAABBGGRR
        uint32_t color = 0xff000000;

        int grey = static_cast<int>(roundf(norm_temp * 255.0f)) & 0xff;
        color = grey << (0 * 8)  // Red
              | grey << (1 * 8)  // Green
              | grey << (2 * 8)  // Blue
              | 0xff << (3 * 8); // Alpha

        pixels_out[idx] = color;
    }
}

bool check_last_cuda_error()
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        loge("CUDA Runtime Error: %s", cudaGetErrorString(err));
        return false;
    }
    return true;
}

int main(int argc, char *argv[])
{
    ApplicationState app_state     = {};
    app_state.alpha                = 0.5f;
    app_state.lighter_temp         = 1.0f;
    app_state.speed_multiplier     = 1;
    app_state.show_settings_window = true;

    app_state.host_pixel_buffer = new uint32_t[WINDOW_WIDTH*WINDOW_HEIGHT];
    if (!app_state.host_pixel_buffer) {
        loge("Failed to allocate the host pixel buffer");
        return 1;
    }
    memset(app_state.host_pixel_buffer, 0, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(uint32_t));
    defer { delete[] app_state.host_pixel_buffer; };

    cudaMalloc(&app_state.device_pixel_buffer, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(uint32_t));
    if (!check_last_cuda_error() || !app_state.device_pixel_buffer) {
        loge("Failed to allocate device pixel buffer");
        return 1;
    }
    cudaMemset(app_state.device_pixel_buffer, 0, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(uint32_t));
    defer { cudaFree(app_state.device_pixel_buffer); };

    cudaMalloc(&app_state.primary_temp_buffer, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(float));
    if (!check_last_cuda_error() || !app_state.primary_temp_buffer) {
        loge("Failed to allocate primary temperature buffer");
        return 1;
    }
    cudaMemset(app_state.primary_temp_buffer, 0, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(float));
    defer { cudaFree(app_state.primary_temp_buffer); };

    cudaMalloc(&app_state.secondary_temp_buffer, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(float));
    if (!check_last_cuda_error() || !app_state.secondary_temp_buffer) {
        loge("Failed to allocate secondary temperature buffer");
        return 1;
    }
    cudaMemset(app_state.secondary_temp_buffer, 0, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(float));
    defer { cudaFree(app_state.secondary_temp_buffer); };
    cudaDeviceSynchronize();

    Display *display = XOpenDisplay(0);
    if (!display) {
        loge("Failed to open display");
        return 1;
    }
    defer { XCloseDisplay(display); };

    Window window = XCreateSimpleWindow(
        display, XDefaultRootWindow(display), 0, 0,
        WINDOW_WIDTH, WINDOW_HEIGHT, 0, 0, 0);
    defer { XDestroyWindow(display, window); };

    Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", false);
    XSetWMProtocols(display, window, &wm_delete_window, 1);

    GLXContext gl_context = {};
    {
        int n = 0;
        XVisualInfo *visual_info = XGetVisualInfo(display, 0, 0, &n);
        gl_context = glXCreateContext(display, visual_info, NULL, GL_TRUE);
    }
    if (!gl_context) {
        loge("Failed to create OpenGL context");
        return 1;
    }
    defer { glXDestroyContext(display, gl_context); };

    if (!glXMakeCurrent(display, window, gl_context)) {
        loge("Failed to attach the OpenGL context to the window");
        return 1;
    }

    logi("GL Renderer:  %s", glGetString(GL_RENDERER));
    logi("GL Version:   %s", glGetString(GL_VERSION));
    logi("GLSL Version: %s", glGetString(GL_SHADING_LANGUAGE_VERSION));

    glGenTextures(1, &app_state.texture_handle);
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    XSelectInput(display, window,
        ExposureMask|KeyPressMask|KeyReleaseMask|
        PointerMotionMask|ButtonPressMask|ButtonReleaseMask);
    XStoreName(display, window, "CUDA Heat Conduction Demo");
    XMapWindow(display, window);

    if(!IMGUI_CHECKVERSION()) {
        loge("IMGUI_CHECKVERSION failed");
        return 1;
    }

    if (!ImGui::CreateContext()) {
        loge("Failed to create ImGui context");
        return 1;
    }
    defer { ImGui::DestroyContext(); };

    if (!ImGui_ImplXlib_InitForOpenGL(display, window, gl_context)) {
        loge("Failed to initialize ImGui for Xlib");
        return 1;
    }
    defer { ImGui_ImplXlib_Shutdown(); };

    if (!ImGui_ImplOpenGL2_Init()) {
        loge("Failed to initialize ImGui for OpenGL2");
        return 1;
    }
    defer { ImGui_ImplOpenGL2_Shutdown(); };

    ImGui::StyleColorsDark();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = NULL;

    cudaDeviceProp device_properties;
    if (cudaGetDeviceProperties(&device_properties, 0) != cudaSuccess) {
        loge("Failed to read device properties");
        return 1;
    }
    logi("GPU Name: %s", device_properties.name);
    logi("Max. threads per block: %d", device_properties.maxThreadsPerBlock);

    app_state.threads_per_block = device_properties.maxThreadsPerBlock;
    app_state.number_of_blocks = static_cast<int>(ceilf(static_cast<float>(WINDOW_HEIGHT*WINDOW_WIDTH)/app_state.threads_per_block));

    auto begin_frame_time = std::chrono::high_resolution_clock::now();
    auto simulation_timer = std::chrono::high_resolution_clock::now();

    while (!app_state.should_close) {
        while (XPending(display) > 0) {
            XEvent event = {};
            XNextEvent(display, &event);
            ImGui_ImplXlib_ProcessEvent(&event);
            switch (event.type)
            {
            case KeyPress: {
                XKeyEvent *key_event = (XKeyEvent *)&event;
                KeySym key_sym = XLookupKeysym(key_event, 0);
                if (key_sym == XK_F1) {
                    app_state.show_settings_window = !app_state.show_settings_window;
                }
            } break;

            case ClientMessage: {
                if ((Atom)event.xclient.data.l[0] == wm_delete_window) {
                    app_state.should_close = true;
                }
            } break;

            case ButtonPress:
            {
                XButtonPressedEvent *button_event = reinterpret_cast<XButtonPressedEvent *>(&event);
                if (button_event->button == Button3) {
                    float t = 1.0f;
                    XWindowAttributes window_attributes = {};
                    XGetWindowAttributes(display, window, &window_attributes);
                    int x = static_cast<int>(roundf(static_cast<float>(button_event->x)/window_attributes.width*WINDOW_WIDTH));
                    int y = static_cast<int>(roundf(static_cast<float>(button_event->y)/window_attributes.height*WINDOW_HEIGHT));
                    int idx = y * WINDOW_WIDTH + x;
                    cudaMemcpy(app_state.primary_temp_buffer+idx, &t, sizeof(float), cudaMemcpyHostToDevice);
                }
            } break;

            case Expose: {
                XWindowAttributes window_attributes = {};
                XGetWindowAttributes(display, window, &window_attributes);
                glViewport(0, 0, window_attributes.width, window_attributes.height);
            } break;

            default:
                break;
            }
        }

        {
            auto now = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(now - simulation_timer).count();
            simulation_timer = now;

            if (!app_state.simulation_paused) {
                for (int i = 0; i < app_state.speed_multiplier; ++i) {
                    heat_conduction_kernel<<<app_state.number_of_blocks, app_state.threads_per_block>>>(
                        WINDOW_WIDTH, WINDOW_HEIGHT, app_state.alpha * dt,
                        app_state.primary_temp_buffer, app_state.secondary_temp_buffer);
                    cudaDeviceSynchronize();
                    std::swap(app_state.primary_temp_buffer, app_state.secondary_temp_buffer);
                }
            }

            color_kernel<<<app_state.number_of_blocks, app_state.threads_per_block>>>(
                WINDOW_WIDTH, WINDOW_HEIGHT, app_state.lighter_temp,
                app_state.primary_temp_buffer, app_state.device_pixel_buffer);
            cudaDeviceSynchronize();

            cudaMemcpy(app_state.host_pixel_buffer, app_state.device_pixel_buffer,
                       WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            glBindTexture(GL_TEXTURE_2D, app_state.texture_handle);

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, app_state.host_pixel_buffer);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

            glEnable(GL_TEXTURE_2D);

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glMatrixMode(GL_TEXTURE);
            glLoadIdentity();

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();

            glBegin(GL_TRIANGLES);
            glTexCoord2i(0, 1);
            glVertex2f(-1.0f, -1.0f);
            glTexCoord2i(0,  0);
            glVertex2f(-1.0f,  1.0f);
            glTexCoord2i(1, 0);
            glVertex2f(1.0f,  1.0f);
            glTexCoord2i(0, 1);
            glVertex2f(-1.0f, -1.0f);
            glTexCoord2i(1, 0);
            glVertex2f(1.0f,  1.0f);
            glTexCoord2i( 1, 1);
            glVertex2f(1.0f, -1.0f);
            glEnd();

            ImGui_ImplOpenGL2_NewFrame();
            ImGui_ImplXlib_NewFrame();
            ImGui::NewFrame();
            if (app_state.show_settings_window) {
                ImGui::Begin("Settings");
                if (ImGui::Button("Reset")) {
                    cudaMemset(app_state.primary_temp_buffer, 0, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(float));
                }
                if (app_state.simulation_paused) {
                    if(ImGui::Button("Play")) {
                        app_state.simulation_paused = !app_state.simulation_paused;
                    }
                } else {
                    if(ImGui::Button("Pause")) {
                        app_state.simulation_paused = !app_state.simulation_paused;
                    }
                }
                ImGui::SliderInt("Speed multiplier", &app_state.speed_multiplier, 1, 10);
                ImGui::SliderFloat("Alpha", &app_state.alpha, 0.1f, 0.9f, "%.01f");

                ImGui::Text("Stats:");
                ImGui::Text("%.02f FPS", app_state.fps);
                ImGui::Text("%d Threads per block", app_state.threads_per_block);
                ImGui::Text("%d Blocks", app_state.number_of_blocks);
                ImGui::End();
            }
            ImGui::Render();
            ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

            glXSwapBuffers(display, window);
        }

        // Enforce frame rate
        auto frame_time_elapsed = std::chrono::high_resolution_clock::now() - begin_frame_time;
        if (frame_time_elapsed < TARGET_FRAME_TIME) {
            std::this_thread::sleep_for(TARGET_FRAME_TIME - frame_time_elapsed);
            while (frame_time_elapsed < TARGET_FRAME_TIME) {
                frame_time_elapsed = std::chrono::high_resolution_clock::now() - begin_frame_time;
            }
        }
        app_state.fps = 1.0f / std::chrono::duration<float>(frame_time_elapsed).count();
        begin_frame_time = std::chrono::high_resolution_clock::now();
    }

    return 0;
}
