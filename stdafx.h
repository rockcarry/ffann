#ifndef __STDAFX_H__
#define __STDAFX_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef WIN32
typedef unsigned int   uint32_t;
typedef   signed int    int32_t;
typedef unsigned short uint16_t;
typedef   signed short  int16_t;
typedef unsigned char  uint8_t;
typedef   signed char   int8_t;
#else
#include <stdint.h>
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef ALIGN
#define ALIGN(x, y) (((x) + (y) - 1) & ~((y) - 1))
#endif

uint32_t get_tick_count(void);

#endif

