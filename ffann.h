#ifndef __FFANN_H__
#define __FFANN_H__

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
void    ann_dump    (ANN *ann);
#define ann_output(ann) ((ann)->nodeval[(ann)->layer_num-1])

#endif
