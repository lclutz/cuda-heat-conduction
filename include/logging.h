#pragma once

#include <stdio.h>

static void prefix_println(const char *prefix, const char *fmt, ...)
{
    fprintf(stderr, "%s", prefix);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

#define logi(...)  prefix_println("INFO: ",  __VA_ARGS__)
#define logw(...)  prefix_println("WARN: ",  __VA_ARGS__)
#define loge(...)  prefix_println("ERROR: ", __VA_ARGS__)
