#ifndef __MATRIX_H__
#define __MATRIX_H__

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

#endif

