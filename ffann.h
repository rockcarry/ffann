#ifndef __FFANN_H__
#define __FFANN_H__

#include <stdint.h>
#include "matrix.h"

enum {
    FFANN_ACTIVATE_SIGMOID,
    FFANN_ACTIVATE_RELU,
    FFANN_ACTIVATE_LEAKY,
    FFANN_ACTIVATE_SOFTMAX,
};

#define ANN_MAX_LAYER  8
typedef struct {
    int32_t layer_num;
    int32_t node_num_max;
    int32_t node_num_list[ANN_MAX_LAYER];
    int32_t activate_list[ANN_MAX_LAYER];
    float  *nodey [ANN_MAX_LAYER];
    float  *nodez [ANN_MAX_LAYER];
    float  *bias  [ANN_MAX_LAYER];
    MATRIX  weight[ANN_MAX_LAYER];
    MATRIX *delta;
    MATRIX *loss;
    MATRIX *dw;
} ANN;

ANN*  ann_create  (int layer_num, int *node_num_list, int *activate_list);
ANN*  ann_load    (char *file);
void  ann_destroy (ANN *ann);
void  ann_forward (ANN *ann, float *input);
void  ann_backward(ANN *ann, float *target, float rate);
float*ann_output  (ANN *ann, int   *num);
float ann_loss    (ANN *ann, float *target);
void  ann_save    (ANN *ann, char  *file);
void  ann_dump    (ANN *ann, char  *file);

#endif

