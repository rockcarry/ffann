#ifndef __FFANN_H__
#define __FFANN_H__

#include "matrix.h"

#define ANN_MAX_LAYER  10
typedef struct {
    int     layer_num;
    int     node_num_list[ANN_MAX_LAYER];
    float  *nodeval[ANN_MAX_LAYER];
    float  *biasval[ANN_MAX_LAYER];
    MATRIX  wmatrix[ANN_MAX_LAYER];
    int     node_num_max;

    MATRIX *delta;
    MATRIX *error;
    MATRIX *dw;
} ANN;

ANN*  ann_create  (int layer_num, int *node_num_list);
ANN*  ann_load    (char *file);
void  ann_destroy (ANN *ann);
void  ann_forward (ANN *ann, float *input);
void  ann_backward(ANN *ann, float *target, float rate);
float ann_error   (ANN *ann, float *target);
void  ann_save    (ANN *ann, char *file);
void  ann_dump    (ANN *ann, char *file);

#define ann_output(ann) ((ann)->nodeval[(ann)->layer_num - 1])

#endif

