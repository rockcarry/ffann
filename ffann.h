#ifndef __FFANN_H__
#define __FFANN_H__

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


#include "matrix.h"
#define ANN_MAX_LAYER  10
typedef struct {
    int     layer_num;
    int     node_num_list[ANN_MAX_LAYER];
    int     bias_flg_list[ANN_MAX_LAYER];
    double *nodeval[ANN_MAX_LAYER - 0];
    MATRIX  wmatrix[ANN_MAX_LAYER - 1];
    int     node_num_max;

    MATRIX *delta;
    MATRIX *dtnew;
    MATRIX *copy;
    MATRIX *dw;
} ANN;

ANN*    ann_create  (int laynum, int *node_num_list, int *bias_flg_list);
void    ann_destroy (ANN *ann);
void    ann_forward (ANN *ann, double *input);
void    ann_backward(ANN *ann, double *target, double rate);
double  ann_error   (ANN *ann, double *target);
ANN*    ann_load    (char*file);
void    ann_save    (ANN *ann, char *file);
void    ann_dump    (ANN *ann, char *file);
#define ann_output(ann) ((ann)->nodeval[(ann)->layer_num-1])

#endif

