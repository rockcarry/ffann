#include "stdafx.h"
#include "matrix.h"

MATRIX* matrix_create(int rows, int cols)
{
    MATRIX *matrix = malloc(sizeof(MATRIX) + rows * cols * sizeof(double));
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