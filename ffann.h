#ifndef __FFANN_H__
#define __FFANN_H__

typedef struct {
    int     rows;
    int     cols;
    double *data;
} MATRIX;

MATRIX* matrix_create  (int rows, int cols);
void    matrix_destroy (MATRIX *m);
void    matrix_multiply(MATRIX *mr, MATRIX *m1, MATRIX *m2 );
void    matrix_adjust  (MATRIX *wt, MATRIX *dw, double rate);
void    matrix_print   (MATRIX *m);

#define ANN_MAX_LAYER  10
typedef struct {
    int     layer_num;
    int     node_num_list[ANN_MAX_LAYER];
    int     bias_flg_list[ANN_MAX_LAYER];
    double *nodeva[ANN_MAX_LAYER - 0];
    MATRIX *matrix[ANN_MAX_LAYER - 1];
    int     node_num_max;
} ANN;

ANN*    ann_create  (int laynum, int *node_num_list, int *bias_flg_list);
void    ann_destroy (ANN *ann);
void    ann_forward (ANN *ann, double *input);
void    ann_backward(ANN *ann, double *target, double rate);
double  ann_total_loss  (ANN *ann, double *target);
void    ann_print_node  (ANN *ann, int start, int end);
void    ann_print_matrix(ANN *ann, int start, int end);
#define ann_print_output(ann) ann_print_node((ann), (ann)->layer_num - 1, (ann)->layer_num - 1)

typedef struct {
    int input_num;
    int output_num;
    double *datain;
    double *dataout;
} SAMPLE;

SAMPLE* sample_create (int inputn, int outputn);
void    sample_destroy(SAMPLE *sample);

#endif
