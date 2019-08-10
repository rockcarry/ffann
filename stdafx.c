#include "stdafx.h"

#ifdef WIN32
#include <windows.h>
#else
#include <time.h>
#endif

uint32_t get_tick_count(void)
{
#ifdef WIN32
    return GetTickCount();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
#endif
}

