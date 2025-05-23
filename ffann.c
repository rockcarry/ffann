#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "ffann.h"

static void activate_forward(int type, float *in, float *out, int num)
{
    float max, sum;
    int   i;
    switch (type) {
    case FFANN_ACTIVATE_SIGMOID:
        for (i = 0; i < num; i++) out[i] = 1 / (1 + expf(-1.0 * in[i]));
        break;
    case FFANN_ACTIVATE_RELU:
        for (i = 0; i < num; i++) out[i] = in[i] > 0 ? in[i] : 0;
        break;
    case FFANN_ACTIVATE_LEAKY:
        for (i = 0; i < num; i++) out[i] = in[i] > 0 ? in[i] : 0.01 * in[i];
        break;
    case FFANN_ACTIVATE_SOFTMAX:
        max = 0, sum = 0;
        for (i = 0; i < num; i++) { max     = MAX(max, in[i]); }
        for (i = 0; i < num; i++) { out[i]  = expf(in[i] - max); sum += out[i]; }
        for (i = 0; i < num; i++) { out[i] /= sum; }
        break;
    }
}

static void activate_backward(int type, float *nodey, float *nodez, float *loss, float *delta, int num)
{
    int i;
    switch (type) {
    case FFANN_ACTIVATE_SIGMOID:
        for (i = 0; i < num; i++) delta[i] = loss[i] * nodez[i] * (1 - nodez[i]);
        break;
    case FFANN_ACTIVATE_RELU:
        for (i = 0; i < num; i++) delta[i] = loss[i] * (nodey[i] > 0 ? 1 : 0);
        break;
    case FFANN_ACTIVATE_LEAKY:
        for (i = 0; i < num; i++) delta[i] = loss[i] * (nodey[i] > 0 ? 1 : 0.01);
        break;
    case FFANN_ACTIVATE_SOFTMAX:
        for (i = 0; i < num; i++) delta[i] = loss[i] * 1;
        break;
    }
}

ANN* ann_create(int layer_num, int *node_num_list, int *activate_list)
{
    int n, i;
    if (layer_num < 2 || !node_num_list) {
        printf("ann_create: invald laynum or node_num_list !\n");
        return NULL;
    }

    layer_num = MIN(layer_num, ANN_MAX_LAYER);
    for (n = node_num_list[0] * 2, i = 1; i < layer_num; i++) {
        n += node_num_list[i] * 3; // nodey & nodez & biases
        n += node_num_list[i - 1] * node_num_list[i - 0]; // weights
    }

    ANN *ann = calloc(1, sizeof(ANN) + n * sizeof(float));
    if (!ann) { printf("ann_create: failed to allocate memory !\n"); return NULL; }

    ann->layer_num = layer_num;
    if (node_num_list) memcpy(ann->node_num_list, node_num_list, layer_num * sizeof(int));
    if (activate_list) memcpy(ann->activate_list, activate_list, layer_num * sizeof(int));

    float *pbuf = (float*)(ann + 1);
    for (i = 0; i < layer_num; i++) {
        ann->nodey[i] = pbuf; pbuf += node_num_list[i];
        ann->nodez[i] = pbuf; pbuf += node_num_list[i];
        ann->node_num_max = MAX(ann->node_num_max, node_num_list[i]);
        if (i > 0) {
            ann->bias[i] = pbuf;
            for (n = 0; n < node_num_list[i]; n++) *pbuf++ = ann->activate_list[i] == 0 ? 0 : 0.01; // biases init
            ann->weight[i].rows = node_num_list[i - 1];
            ann->weight[i].cols = node_num_list[i - 0];
            ann->weight[i].data = pbuf;
            for (n = 0; n < ann->weight[i].rows * ann->weight[i].cols; n++) { // xavier weights init
                float v = sqrtf(6.0 / (ann->weight[i].rows + ann->weight[i].cols));
                pbuf[n] = v * (float)((rand() % RAND_MAX) - (RAND_MAX / 2.0)) / (RAND_MAX / 2.0);
            }
            pbuf += n;
        }
    }
    return ann;
}

void ann_destroy(ANN *ann)
{
    if (!ann) return;
    matrix_destroy(ann->delta);
    matrix_destroy(ann->loss );
    matrix_destroy(ann->dw   );
    free(ann);
}

void ann_forward(ANN *ann, float *input)
{
    MATRIX mi = {1}, mo = {1}, mb = {1};
    int    i;
    if (!ann || !input) {
        printf("ann_forward: invalid ann or input !\n");
        return;
    }

    memcpy(ann->nodez[0], input, ann->node_num_list[0] * sizeof(float)); // copy input to ann input layer

    for (i = 1; i < ann->layer_num; i++) { // for each layer do forward inference
        mi.cols = ann->node_num_list[i - 1];
        mi.data = ann->nodez[i - 1];
        mo.cols = ann->node_num_list[i - 0];
        mo.data = ann->nodey[i - 0];
        mb.cols = ann->node_num_list[i - 0];
        mb.data = ann->bias [i - 0];
        matrix_multiply(&mo, &mi, &ann->weight[i]); // MO = MI * W
        matrix_adjust  (&mo, &mb, 1); // MO = MO + MB
        activate_forward(ann->activate_list[i], ann->nodey[i], ann->nodez[i], ann->node_num_list[i]); // nodez = activate(nodey)
    }
}

void ann_backward(ANN *ann, float *target, float rate)
{
    MATRIX mat = { 1, 1 };
    int    i, j;

    if (!ann || ann->layer_num < 2 || !target) {
        printf("ann_backward: invalid ann or target !\n");
        return;
    }

    if (!ann->delta) ann->delta = matrix_create(1, ann->node_num_max);
    if (!ann->loss ) ann->loss  = matrix_create(ann->node_num_max, 1);
    if (!ann->dw   ) ann->dw    = matrix_create(ann->node_num_max, ann->node_num_max);

    // calculate output layer loss
    float *nodey = ann->nodey[ann->layer_num - 1];
    float *nodez = ann->nodez[ann->layer_num - 1];
    for (j = 0; j < ann->node_num_list[ann->layer_num - 1]; j++) ann->loss->data[j] = (nodez[j] - target[j]);

    for (i = ann->layer_num - 1; i > 0; i--) {
        nodey = ann->nodey[i], nodez = ann->nodez[i];
        activate_backward(ann->activate_list[i], nodey, nodez, ann->loss->data, ann->delta->data, ann->node_num_list[i]); // calculate current layer deltas

        ann->loss->rows = ann->weight[i].rows;
        mat.rows        = ann->weight[i].cols;
        mat.cols        = 1;
        mat.data        = ann->delta->data;
        matrix_multiply(ann->loss, &ann->weight[i], &mat); // calculate current layer loss

        // calculate weight gradient and update weights
        ann->dw->rows    = ann->weight[i].rows;
        ann->dw->cols    = ann->weight[i].cols;
        mat.rows         = ann->dw->rows;
        mat.data         = ann->nodez[i - 1];
        ann->delta->cols = ann->dw->cols;
        matrix_multiply(ann->dw, &mat, ann->delta);
        matrix_adjust(&ann->weight[i], ann->dw, -rate);

        // calculate bias gradient and update biases
        mat.rows = 1, mat.cols = ann->delta->cols, mat.data = ann->bias[i];
        matrix_adjust(&mat, ann->delta, -rate);
    }
}

float *ann_output(ANN *ann, int *num)
{
    if (!ann) return NULL;
    if (num) *num = ann->node_num_list[ann->layer_num - 1];
    return ann->nodez[ann->layer_num - 1];
}

float ann_loss(ANN *ann, float *target)
{
    float loss = 0;
    int   i;
    if (!ann) return 0;
    switch (ann->activate_list[ann->layer_num - 1]) {
    case FFANN_ACTIVATE_SOFTMAX: // for softmax output we use Cross-Entroy Loss
        for (i = 0; i < ann->node_num_list[ann->layer_num - 1]; i++) {
            loss -= target[i] * logf(ann->nodez[ann->layer_num - 1][i] + 1e-10);
        }
        break;
    default:  // for others we use Mean Squared Error Loss
        for (i = 0; i < ann->node_num_list[ann->layer_num - 1]; i++) {
            loss += 0.5 * pow(target[i] - ann->nodez[ann->layer_num - 1][i], 2);
        }
        break;
    }
    return loss;
}

ANN* ann_load(char *file)
{
    int filesize, ret, i;
    FILE * fp = fopen(file, "rb");
    if (!fp) { printf("ann_load: failed to open file %s !\n", file); return NULL; }

    (void)ret;
    fseek(fp, 0, SEEK_END);
    filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    ANN *ann = malloc(filesize);
    if (!ann) { printf("ann_load: failed to allocate memory !\n"); goto done; }

    ret = fread(ann, 1, filesize, fp);
    float *pdata = (float*)(ann + 1);
    for (i = 0; i < ann->layer_num; i++) {
        ann->nodey[i] = pdata; pdata += ann->node_num_list[i];
        ann->nodez[i] = pdata; pdata += ann->node_num_list[i];
        if (i > 0) {
            ann->bias[i] = pdata; pdata += ann->node_num_list[i];
            ann->weight[i].data = pdata; pdata += ann->weight[i].rows * ann->weight[i].cols;
        }
    }
    ann->delta = ann->loss = ann->dw = NULL;

done:
    fclose(fp);
    return ann;
}

void ann_save(ANN *ann, char *file)
{
    FILE *fp = NULL;
    int   n, i;
    if (!ann || !file) { printf("ann_save: invalid samples or file !\n"); return; }
    for (n = ann->node_num_list[0] * 2, i = 1; i<ann->layer_num; i++) {
        n += ann->node_num_list[i] * 3;
        n += ann->node_num_list[i - 1] * ann->node_num_list[i - 0];
    }
    fp = fopen(file, "wb");
    if (!fp) { printf("ann_save: failed to open file %s !\n", file); return; }
    fwrite(ann, 1, sizeof(ANN) + n * sizeof(float), fp);
    fclose(fp);
}

void ann_dump(ANN *ann, char *file)
{
    FILE *fp = stdout;
    int i, j;
    if (!ann) return;
    if (file) fp = fopen(file, "wb");
    if (fp) {
        fprintf(fp, "dump ann info:\n");
        fprintf(fp, "layer_num: %d\n", ann->layer_num);
        fprintf(fp, "node_num_list: ");
        for (i = 0; i < ann->layer_num; i++) {
            fprintf(fp, "%d ", ann->node_num_list[i]);
        }
        fprintf(fp, "\n\n");

        for (i = 0; i < ann->layer_num; i++) {
            if (i > 0) {
                fprintf(fp, "weight_%d:", i);
                matrix_dump(&ann->weight[i], fp);
                fprintf(fp, "bias___%d: ", i);
                for (j = 0; j < ann->node_num_list[i]; j++) {
                    fprintf(fp, "%8.5f ", ann->bias[i][j]);
                }
                fprintf(fp, "\n\n");
            }
            fprintf(fp, "nodey__%d: ", i);
            for (j = 0; j < ann->node_num_list[i]; j++) {
                fprintf(fp, "%8.5f ", ann->nodey[i][j]);
            }
            fprintf(fp, "\n\n");
            fprintf(fp, "nodez__%d: ", i);
            for (j = 0; j < ann->node_num_list[i]; j++) {
                fprintf(fp, "%8.5f ", ann->nodez[i][j]);
            }
            fprintf(fp, "\n\n");
        }
        fprintf(fp, "\n");
        if (file) fclose(fp);
    }
}
