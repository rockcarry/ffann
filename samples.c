#include "stdafx.h"
#include "samples.h"

SAMPLES* samples_create(int sampn, int inputn, int outputn)
{
    SAMPLES *samples = malloc(sizeof(SAMPLES) + sampn * (inputn + outputn) * sizeof(double));
    if (samples) {
        samples->num_samples = sampn;
        samples->num_input   = inputn;
        samples->num_output  = outputn;
        samples->data        = (double*)((uint8_t*)samples + sizeof(SAMPLES));
    } else {
        printf("samples_create: failed to allocate memory !\n");
    }
    return samples;
}

void samples_destroy(SAMPLES *samples)
{
    if (samples) free(samples);
}

SAMPLES* samples_load(char *file)
{
    SAMPLES *samples = NULL;
    FILE    *fp      = NULL;
    int      filesize;
    fp = fopen(file, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        filesize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        samples  = malloc(filesize);
        if (samples) {
            fread(samples, 1, filesize, fp);
            samples->data = (double*)((uint8_t*)samples + sizeof(SAMPLES));
        } else {
            printf("samples_load: failed to allocate memory !\n");
        }
        fclose(fp);
    } else {
        printf("samples_load: failed to open file %s !\n", file);
    }
    return samples;
}

void samples_save(SAMPLES *samples, char *file)
{
    FILE *fp = NULL;
    int   datasize;
    if (!samples || !file) {
        printf("samples_save: invalid samples or file !\n");
        return;
    }
    datasize = samples->num_samples * (samples->num_input + samples->num_output);
    fp = fopen(file, "wb");
    if (fp) {
        fwrite(samples, 1, sizeof(SAMPLES) + datasize * sizeof(double), fp);
        fclose(fp);
    } else {
        printf("samples_save: failed to open file %s !\n", file);
    }
}

double* samples_get_input(SAMPLES *samples, int idx)
{
    if (!samples) return NULL;
    return samples->data + idx * (samples->num_input + samples->num_output);
}

double* samples_get_output(SAMPLES *samples, int idx)
{
    if (!samples) return NULL;
    return samples->data + idx * (samples->num_input + samples->num_output) + samples->num_input;
}


