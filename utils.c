#include <stdint.h>
#include <time.h>
#include "utils.h"

uint32_t get_timestamp32_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
