#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdint.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

uint32_t get_timestamp32_ms(void);

#endif

