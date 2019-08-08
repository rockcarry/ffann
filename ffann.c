#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define ANN_MAX_LAYER 10

typedef struct {
    int     rows;
    int     cols;
    double *data;
} MATRIX;

typedef struct {
    int     layer_num;
    int     node_num_list[ANN_MAX_LAYER];
    int     bias_flg_list[ANN_MAX_LAYER];
    double *nodeva[ANN_MAX_LAYER - 0];
    MATRIX *matrix[ANN_MAX_LAYER - 1];
    int     node_num_max;
} ANN;

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

MATRIX* matrix_create(int rows, int cols)
{
    MATRIX *matrix = calloc(1, sizeof(MATRIX) + rows * cols * sizeof(double));
    if (!matrix) {
        printf("matrix_create: failed to allocate memory !\n");
        return NULL;
    }
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (double*)((uint8_t*)matrix + sizeof(MATRIX));
    return matrix;
}

void matrix_destroy(MATRIX *matrix)
{
    if (matrix) free(matrix);
}

void matrix_multiply(MATRIX *mr, MATRIX *m1, MATRIX *m2)
{
    int     r, c, n;
    double *d1, *d2, *dr;
    if (!mr || !m1 || !m2) {
        printf("matrix_multiply: invalid mr, m1 or m2 !\n");
        return;
    }
    if (m1->cols != m2->rows || mr->rows != m1->rows || mr->cols != m2->cols) {
        printf("matrix_multiply: invalid rows or cols !\n");
        return;
    }
    d1 = m1->data;
    d2 = m2->data;
    dr = mr->data;
    for (r=0; r<mr->rows; r++) {
        for (c=0; c<mr->cols; c++) {
            for (dr[c]=0,n=0; n<m1->cols; n++) {
                dr[c] += d1[n] * d2[c+m2->cols*n];
            }
        }
        d1 += m1->cols;
        dr += mr->cols;
    }
}

void matrix_add(MATRIX *m1, MATRIX *m2)
{
    int i, n;
    if (!m1 || !m2) {
        printf("matrix_add: invalid m1 or m2 !\n");
        return;
    }
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        printf("matrix_add: invalid rows or cols !\n");
        return;
    }
    for (i=0,n=m1->rows*m1->cols; i<n; i++) {
        m1->data[i] += m2->data[i];
    }
}

void matrix_scale(MATRIX *matrix, double scale)
{
    int i, n;
    if (!matrix) {
        printf("matrix_scale: invalid matrix !\n");
        return;
    }
    for (i=0,n=matrix->rows*matrix->cols; i<n; i++) {
        matrix->data[i] *= scale;
    }
}

void matrix_print(MATRIX *m)
{
    int r, c;
    printf("\n");
    for (r=0; r<m->rows; r++) {
        for (c=0; c<m->cols; c++) {
            printf("%-8.3lf ", m->data[c + r * m->cols]);
        }
        printf("\n");
    }
    printf("\n");
}

ANN* ann_create(int laynum, int *node_num_list, int *bias_flg_list)
{
    double *bufa;
    int     n, i;
    if (laynum < 2 || !node_num_list) {
        printf("ann_create: invald laynum or node_num_list !\n");
        return NULL;
    }

    laynum = laynum < ANN_MAX_LAYER ? laynum : ANN_MAX_LAYER;
    for (n=0,i=0; i<laynum; i++) n += node_num_list[i];
    ANN *ann = calloc(1, sizeof(ANN) + n * sizeof(double));
    if (!ann) {
        printf("ann_create: failed to allocate memory !\n");
        return NULL;
    }

    ann->layer_num = laynum;
    if (node_num_list) memcpy(ann->node_num_list, node_num_list, laynum * sizeof(int));
    if (bias_flg_list) memcpy(ann->bias_flg_list, bias_flg_list, laynum * sizeof(int));

    bufa = (double*)((uint8_t*)ann + sizeof(ANN));
    for (i=0; i<laynum; i++) {
        ann->nodeva[i] = bufa;
        bufa += node_num_list[i];
        ann->node_num_max = ann->node_num_max > node_num_list[i] ? ann->node_num_max : node_num_list[i];
    }

    for (i=0; i<laynum-1; i++) ann->matrix[i] = matrix_create(node_num_list[i], node_num_list[i+1]);
    return ann;
}

void ann_destroy(ANN *ann)
{
    if (ann) {
        int i;
        for (i=0; i<ann->layer_num-1; i++) {
            matrix_destroy(ann->matrix[i]);
        }
        free(ann);
    }
}

void ann_forward(ANN *ann, double *input, int num)
{
    MATRIX mi, mo;
    int    i, n;

    if (!ann || !input || ann->node_num_list[0] != num) {
        printf("ann_forward: invalid ann, input or num !\n");
        return;
    }

    memcpy(ann->nodeva[0], input, num * sizeof(double));
    for (i=0; i<ann->layer_num-1; i++) {
        mi.rows = 1;
        mi.cols = ann->node_num_list[i+0];
        mi.data = ann->nodeva[i+0];
        mo.rows = 1;
        mo.cols = ann->node_num_list[i+1];
        mo.data = ann->nodeva[i+1];
        matrix_multiply(&mo, &mi, ann->matrix[i]);
        for (n=0; n<mo.cols; n++) mo.data[n] = sigmoid(mo.data[n]);
    }
}

void ann_backward(ANN *ann, double *target, int num, double rate)
{
    MATRIX *delta, *dw, prevo;
    int     i, j;

    if (!ann || ann->layer_num < 2 || !target || ann->node_num_list[ann->layer_num-1] != num) {
        printf("ann_backward: invalid ann, target or num !\n");
        return;
    }

    delta = matrix_create(1, ann->node_num_max);
    dw    = matrix_create(ann->node_num_max, ann->node_num_max);

    for (i=ann->layer_num-2; i>=0; i--) {
        if (i == ann->layer_num-2) {
            for (j=0; j<ann->node_num_list[i+1]; j++) {
                double outa = ann->nodeva[i+1][j];
                delta->data[j] = -1 * (target[j] - outa) * outa * (1 - outa);
            }
        } else {
            for (j=0; j<ann->node_num_list[i+1]; j++) {
                double outa = ann->nodeva[i+1][j];
                double value_err_total;
                MATRIX matrix_err_weight;
                MATRIX matrix_err_total;
                matrix_err_weight.rows = ann->matrix[i+1]->cols;
                matrix_err_weight.cols = 1;
                matrix_err_weight.data = ann->matrix[i+1]->data + ann->matrix[i+1]->cols * j;
                matrix_err_total .rows = 1;
                matrix_err_total .cols = 1;
                matrix_err_total .data = &value_err_total;
                matrix_multiply(&matrix_err_total, delta, &matrix_err_weight);
                delta->data[j] = -1 * value_err_total * outa * (1 - outa);
            }
        }
        delta->cols = ann->node_num_list[i+1];

        prevo.rows = ann->node_num_list[i];
        prevo.cols = 1;
        prevo.data = ann->nodeva[i];

        dw->rows = ann->node_num_list[i + 0];
        dw->cols = ann->node_num_list[i + 1];
        matrix_multiply(dw, &prevo, delta);
        matrix_scale(dw, -rate);
        matrix_add(ann->matrix[i], dw);
    }
    matrix_destroy(delta);
    matrix_destroy(dw   );
}

void ann_print_node(ANN *ann, int layer_start, int layer_end)
{
    int i, n;
    if (!ann) return;
    for (i=layer_start; i<=layer_end && i>=0 && i<ann->layer_num; i++) {
        printf("\nlayer_%d: ", i);
        for (n=0; n<ann->node_num_list[i]; n++) {
            printf("%-8.3f ", ann->nodeva[i][n]);
        }
        printf("\n");
    }
}

void ann_print_matrix(ANN *ann, int matrix_start, int matrix_end)
{
    int i;
    if (!ann) return;
    for (i=matrix_start; i<=matrix_end && i>=0 && i<ann->layer_num-1; i++) {
        printf("\nmatrix_%d: ", i);
        matrix_print(ann->matrix[i]);
//      printf("\n");
    }
}

int main(void)
{
    if (0) {
        MATRIX *m1 = matrix_create(2, 3);
        MATRIX *m2 = matrix_create(3, 2);
        MATRIX *mr = matrix_create(2, 2);

        m1->data[0] = 1;  m1->data[1] = 2;  m1->data[2] = 3;
        m1->data[3] = 4;  m1->data[4] = 5;  m1->data[5] = 6;

        m2->data[0] = 7;  m2->data[1] = 10;
        m2->data[2] = 8;  m2->data[3] = 11;
        m2->data[4] = 9;  m2->data[5] = 12;

        matrix_multiply(mr, m1, m2);
        matrix_print(m1);
        matrix_print(m2);
        matrix_print(mr);
        matrix_destroy(m1);
        matrix_destroy(m2);
        matrix_destroy(mr);
    }

    if (1) {
        int node_num_list[] = { 2, 5, 1 };
        double data  [] = { 0, 0 };
        double target[] = { 0 };
        ANN *ann = ann_create(3, node_num_list, NULL);
        ann_forward (ann, data  , 2);
        ann_backward(ann, target, 1, 0.2);
        ann_forward (ann, data  , 2);
        ann_backward(ann, target, 1, 0.2);
        ann_print_node  (ann, 0, 2);
        ann_print_matrix(ann, 0, 1);
        ann_destroy(ann);
    }
    return 0;
}
