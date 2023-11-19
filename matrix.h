#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdio.h>

typedef struct {
    int    rows;
    int    cols;
    float *data;
} MATRIX;

MATRIX* matrix_create  (int rows, int cols);
void    matrix_destroy (MATRIX *m);
void    matrix_multiply(MATRIX *mr, MATRIX *m1, MATRIX *m2);
void    matrix_adjust  (MATRIX *wt, MATRIX *dw, float rate);
void    matrix_dump    (MATRIX *m, FILE *fp);

#endif

