#ifndef __SAMPLES_H__
#define __SAMPLES_H__

typedef struct {
    int num_samples;
    int num_input;
    int num_output;
    double *data;
} SAMPLES;

SAMPLES* samples_create    (int sampn, int inputn, int outputn);
void     samples_destroy   (SAMPLES *samples);
SAMPLES* samples_load      (char *file);
void     samples_save      (SAMPLES *samples, char *file);
double*  samples_get_input (SAMPLES *samples, int idx);
double*  samples_get_output(SAMPLES *samples, int idx);

#endif




