#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ffann.h"

static double sigmoid(double x)
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

void matrix_destroy(MATRIX *m)
{
    if (m) free(m);
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

void matrix_adjust(MATRIX *wt, MATRIX *dw, double rate)
{
    int i, n;
    if (!wt || !dw) {
        printf("matrix_adjust: invalid wt or dw !\n");
        return;
    }
    if (wt->rows != dw->rows || wt->cols != dw->cols) {
        printf("matrix_adjust: invalid rows or cols !\n");
        return;
    }
    for (i=0,n=wt->rows*wt->cols; i<n; i++) {
        wt->data[i] -= dw->data[i] * rate;
    }
}

void matrix_print(MATRIX *m)
{
    int r, c;
    printf("\n");
    for (r=0; r<m->rows; r++) {
        for (c=0; c<m->cols; c++) {
            printf("%-8.5lf ", m->data[c + r * m->cols]);
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

    for (i=0; i<laynum-1; i++) {
        ann->matrix[i] = matrix_create(node_num_list[i], node_num_list[i+1]);
        for (n=0; n<ann->matrix[i]->rows*ann->matrix[i]->cols; n++) { // rand init weight matrixs
            ann->matrix[i]->data[n] = (double)((rand() % RAND_MAX) - (RAND_MAX / 2)) / (RAND_MAX / 2);
        }
    }
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

void ann_forward(ANN *ann, double *input)
{
    MATRIX mi, mo;
    int    i, n;

    if (!ann || !input) {
        printf("ann_forward: invalid ann or input !\n");
        return;
    }

    memcpy(ann->nodeva[0], input, ann->node_num_list[0] * sizeof(double));
    if (ann->bias_flg_list[0]) ann->nodeva[0][ann->node_num_list[0] - 1] = ann->bias_flg_list[0];

    for (i=0; i<ann->layer_num-1; i++) {
        mi.rows = 1;
        mi.cols = ann->node_num_list[i+0];
        mi.data = ann->nodeva[i+0];
        mo.rows = 1;
        mo.cols = ann->node_num_list[i+1];
        mo.data = ann->nodeva[i+1];
        matrix_multiply(&mo, &mi, ann->matrix[i]);
        for (n=0; n<mo.cols; n++) mo.data[n] = sigmoid(mo.data[n]);
        if (ann->bias_flg_list[i+1]) ann->nodeva[i+1][ann->node_num_list[i+1] - 1] = ann->bias_flg_list[i+1];
    }
}

void ann_backward(ANN *ann, double *target, double rate)
{
    MATRIX *delta, *dtnew, *dw, prevo;
    MATRIX *copys[ANN_MAX_LAYER];
    int     i, j;

    if (!ann || ann->layer_num < 2 || !target) {
        printf("ann_backward: invalid ann or target !\n");
        return;
    }

    delta = matrix_create(1, ann->node_num_max);
    dtnew = matrix_create(1, ann->node_num_max);
    dw    = matrix_create(ann->node_num_max, ann->node_num_max);
    for (i=0; i<ann->layer_num-1; i++) {
        copys[i] = matrix_create(ann->matrix[i]->rows, ann->matrix[i]->cols);
        memcpy(copys[i]->data, ann->matrix[i]->data, ann->matrix[i]->rows * ann->matrix[i]->cols * sizeof(double));
    }

    for (i=ann->layer_num-2; i>=0; i--) {
        // calculate delta vector
        if (i == ann->layer_num-2) {
            for (j=0; j<ann->node_num_list[i+1]; j++) {
                double outa = ann->nodeva[i+1][j];
                delta->data[j] = -1 * (target[j] - outa) * outa * (1 - outa);
            }
        } else {
            for (j=0; j<ann->node_num_list[i+1]; j++) {
                double outa = ann->nodeva[i+1][j], value_err_total;
                MATRIX matrix_err_weight = { copys[i+1]->cols, 1, copys[i+1]->data + copys[i+1]->cols * j };
                MATRIX matrix_err_total  = { 1, 1, &value_err_total };
                matrix_multiply(&matrix_err_total, delta, &matrix_err_weight);
                dtnew->data[j] = value_err_total * outa * (1 - outa);
            }
            memcpy(delta->data, dtnew->data, ann->node_num_list[i+1] * sizeof(double));
        }
        delta->cols = ann->node_num_list[i+1];

        // calculate prev output vector
        prevo.rows = ann->node_num_list[i];
        prevo.cols = 1;
        prevo.data = ann->nodeva[i];

        dw->rows = ann->node_num_list[i + 0];
        dw->cols = ann->node_num_list[i + 1];
        matrix_multiply(dw, &prevo, delta);
        matrix_adjust  (ann->matrix[i], dw, rate);
    }

    matrix_destroy(delta);
    matrix_destroy(dtnew);
    matrix_destroy(dw   );
    for (i=0; i<ann->layer_num-1; i++) {
        matrix_destroy(copys[i]);
    }
}

void ann_print_node(ANN *ann, int start, int end)
{
    int i, n;
    if (!ann) return;
    for (i=start; i<=end && i>=0 && i<ann->layer_num; i++) {
        printf("\nlayer_%d: ", i);
        for (n=0; n<ann->node_num_list[i]; n++) {
            printf("%-8.5lf ", ann->nodeva[i][n]);
        }
        printf("\n");
    }
}

void ann_print_matrix(ANN *ann, int start, int end)
{
    int i;
    if (!ann) return;
    for (i=start; i<=end && i>=0 && i<ann->layer_num-1; i++) {
        printf("\nmatrix_%d: ", i);
        matrix_print(ann->matrix[i]);
    }
}

double ann_total_loss(ANN *ann, double *target)
{
    double loss = 0;
    int    i;
    if (!ann) return 0;
    for (i=0; i<ann->node_num_list[ann->layer_num-1]; i++) {
        loss += 0.5 * pow(target[i] - ann->nodeva[ann->layer_num-1][i], 2);
    }
    return loss;
}

SAMPLE* sample_create(int inputn, int outputn)
{
    SAMPLE *sample = calloc(1, sizeof(SAMPLE) + (inputn + outputn) * sizeof(double));
    if (sample) {
        sample->input_num  = inputn;
        sample->output_num = outputn;
        sample->datain     = (double*)((uint8_t*)sample + sizeof(SAMPLE));
        sample->dataout    = sample->datain + inputn;
    }
    return sample;
}

void sample_destroy(SAMPLE *sample)
{
    if (sample) free(sample);
}

static uint32_t get_tick_count(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

int main(void)
{
    double rate = 0.5, loss = 1;
    int node_num_list[] = { 3, 5, 1 };
    int bias_flg_list[] = { 1, 1, 0 };
    int i, j, n = 0;
    uint32_t tick;
    ANN    *ann;
    SAMPLE *samples[4];

    ann = ann_create(3, node_num_list, bias_flg_list);
    for (i=0; i<4; i++) {
        samples[i] = sample_create(3, 1);
    }

    samples[0]->datain [0] = 0;
    samples[0]->datain [1] = 0;
    samples[0]->dataout[0] = 0;

    samples[1]->datain [0] = 0;
    samples[1]->datain [1] = 1;
    samples[1]->dataout[0] = 1;

    samples[2]->datain [0] = 1;
    samples[2]->datain [1] = 0;
    samples[2]->dataout[0] = 1;

    samples[3]->datain [0] = 1;
    samples[3]->datain [1] = 1;
    samples[3]->dataout[0] = 1;

    tick = get_tick_count();
    while (loss > 0.000001) {
        loss = 0;
        for (j=0; j<4; j++) {
            ann_forward (ann, samples[j]->datain);
            ann_backward(ann, samples[j]->dataout, rate);
            loss += ann_total_loss(ann, samples[j]->dataout);
        }
        if (get_tick_count() - tick >= 1000) {
            printf("%5d total loss: %lf\n", ++n, loss);
            fflush(stdout);
            tick += 1000;
        }
    }

    for (i=0; i<4; i++) {
        ann_forward(ann, samples[i]->datain);
        ann_print_output(ann);
    }

    ann_destroy(ann);
    for (i=0; i<4; i++) {
        sample_destroy(samples[i]);
    }
    return 0;
}
