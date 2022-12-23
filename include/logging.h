#pragma once

#include <stdio.h>

static void prefixed_println(const char *prefix, const char *fmt, ...)
{
    fprintf(stderr, "%s", prefix);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

#define logi(...)  prefixed_println("INFO: ",  __VA_ARGS__)
#define logw(...)  prefixed_println("WARN: ",  __VA_ARGS__)
#define loge(...)  prefixed_println("ERROR: ", __VA_ARGS__)
