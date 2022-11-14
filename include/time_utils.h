#pragma once

#include <time.h>

timespec get_wall_clock()
{
    timespec result;
    clock_gettime(CLOCK_MONOTONIC, &result);
    return result;
}

float get_seconds_elapsed(timespec start, timespec end)
{
    float delta_seconds = (float) (end.tv_sec - start.tv_sec);
    float delta_nanoseconds = (float) (end.tv_nsec - start.tv_nsec);
    return delta_seconds + delta_nanoseconds * 1.0e-9f;
}
