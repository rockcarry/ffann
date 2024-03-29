#ifndef __FFANN_H__
#define __FFANN_H__

#include "matrix.h"

enum {
    FFANN_ACTIVATE_SIGMOID,
    FFANN_ACTIVATE_RELU,
    FFANN_ACTIVATE_LEAKY,
    FFANN_ACTIVATE_SOFTMAX,
};

#define ANN_MAX_LAYER  10
typedef struct {
    int     layer_num;
    int     node_num_max;
    int     node_num_list[ANN_MAX_LAYER];
    int     activate_list[ANN_MAX_LAYER];
    float  *nodei [ANN_MAX_LAYER];
    float  *nodeo [ANN_MAX_LAYER];
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

