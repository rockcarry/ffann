#include <stdlib.h>
#include "ffann.h"
#include "matrix.h"

MATRIX* matrix_create(int rows, int cols)
{
    MATRIX *matrix = malloc(sizeof(MATRIX) + rows * cols * sizeof(float));
    if (!matrix) { printf("matrix_create: failed to allocate memory !\n"); return NULL; }
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (float*)(matrix + 1);
    return matrix;
}

void matrix_destroy(MATRIX *m)
{
    free(m);
}

void matrix_multiply(MATRIX *mr, MATRIX *m1, MATRIX *m2)
{
    int    r, c, n;
    float *d1, *d2, *dr;
    if (!mr || !m1 || !m2) {
        printf("matrix_multiply: invalid mr, m1 or m2 !\n");
        return;
    }
    if (m1->cols != m2->rows || mr->rows != m1->rows || mr->cols != m2->cols) {
        printf("matrix_multiply: invalid rows or cols ! (%d,%d) (%d,%d) (%d,%d)\n", mr->rows, mr->cols, m1->rows, m1->cols, m2->rows, m2->cols);
        return;
    }
    d1 = m1->data;
    d2 = m2->data;
    dr = mr->data;
    for (r = 0; r < mr->rows; r++) {
        for (c = 0; c < mr->cols; c++) {
            for (dr[c] = 0, n = 0; n < m1->cols; n++) {
                dr[c] += d1[n] * d2[c + m2->cols*n];
            }
        }
        d1 += m1->cols;
        dr += mr->cols;
    }
}

void matrix_adjust(MATRIX *wt, MATRIX *dw, float rate)
{
    int i, n;
    if (!wt || !dw) {
        printf("matrix_adjust: invalid wt or dw !\n");
        return;
    }
    if (wt->rows != dw->rows || wt->cols != dw->cols) {
        printf("matrix_adjust: invalid rows or cols ! (%d,%d) (%d,%d)\n", wt->rows, wt->cols, dw->rows, dw->cols);
        return;
    }
    for (n = wt->rows * wt->cols, i = 0; i < n; i++) {
        wt->data[i] += dw->data[i] * rate;
    }
}

void matrix_dump(MATRIX *m, FILE *fp)
{
    int r, c;
    fp = fp ? fp : stdout;
    fprintf(fp, "\n");
    for (r = 0; r < m->rows; r++) {
        for (c = 0; c < m->cols; c++) {
            fprintf(fp, "%8.5f ", m->data[c + r * m->cols]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
}
