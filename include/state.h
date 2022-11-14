#pragma once

#include <stdint.h>

#include <GL/gl.h>

struct ApplicationState
{
    bool should_close;
    bool show_settings_window;
    bool simulation_paused;

    GLuint   texture_handle;
    uint32_t *host_pixel_buffer;

    uint32_t *device_pixel_buffer;
    float    *primary_temp_buffer;
    float    *secondary_temp_buffer;
    float    alpha;
    float    lighter_temp;
    int      speed_multiplier;

    // Stats
    float fps;
    int threads_per_block;
    int number_of_blocks;
};
