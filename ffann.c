#include <time.h>
#include "stdafx.h"
#include "ffann.h"

static double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

ANN* ann_create(int laynum, int *node_num_list, int *bias_flg_list)
{
    double *pbuf;
    int     n, i;

    if (laynum < 2 || !node_num_list) {
        printf("ann_create: invald laynum or node_num_list !\n");
        return NULL;
    }

    laynum = MIN(laynum, ANN_MAX_LAYER);
    for (n=node_num_list[0],i=1; i<laynum; i++) {
        n += node_num_list[i];
        n += node_num_list[i-1] * node_num_list[i];
    }

    ANN *ann = malloc(sizeof(ANN) + n * sizeof(double));
    if (!ann) {
        printf("ann_create: failed to allocate memory !\n");
        return NULL;
    }

    memset(ann, 0, sizeof(ANN));
    ann->layer_num = laynum;
    if (node_num_list) memcpy(ann->node_num_list, node_num_list, laynum * sizeof(int));
    if (bias_flg_list) memcpy(ann->bias_flg_list, bias_flg_list, laynum * sizeof(int));

    pbuf = (double*)((uint8_t*)ann + sizeof(ANN));
    for (i=0; i<laynum; i++) {
        ann->nodeval[i] = pbuf;
        pbuf += node_num_list[i];
        ann->node_num_max = MAX(ann->node_num_max, node_num_list[i]);
    }

    for (i=0; i<laynum-1; i++) {
        ann->wmatrix[i].rows = node_num_list[i + 0];
        ann->wmatrix[i].cols = node_num_list[i + 1];
        ann->wmatrix[i].data = pbuf;
        pbuf += ann->wmatrix[i].rows * ann->wmatrix[i].cols;
        for (n=0; n<ann->wmatrix[i].rows*ann->wmatrix[i].cols; n++) { // rand init weight matrixs
            ann->wmatrix[i].data[n] = (double)((rand() % RAND_MAX) - (RAND_MAX / 2)) / (RAND_MAX / 2);
        }
    }
    return ann;
}

void ann_destroy(ANN *ann)
{
    if (ann) {
        matrix_destroy(ann->delta);
        matrix_destroy(ann->dtnew);
        matrix_destroy(ann->copy );
        matrix_destroy(ann->dw   );
        free(ann);
    }
}

void ann_forward(ANN *ann, double *input)
{
    MATRIX mi = {1}, mo = {1};
    int    i, n;

    if (!ann || !input) {
        printf("ann_forward: invalid ann or input !\n");
        return;
    }

    memcpy(ann->nodeval[0], input, ann->node_num_list[0] * sizeof(double));
    if (ann->bias_flg_list[0]) ann->nodeval[0][ann->node_num_list[0] - 1] = ann->bias_flg_list[0];

    for (i=0; i<ann->layer_num-1; i++) {
        mi.cols = ann->node_num_list[i+0];
        mi.data = ann->nodeval[i+0];
        mo.cols = ann->node_num_list[i+1];
        mo.data = ann->nodeval[i+1];
        matrix_multiply(&mo, &mi, &ann->wmatrix[i]);
        for (n=0; n<mo.cols; n++) mo.data[n] = sigmoid(mo.data[n]);
        if (ann->bias_flg_list[i+1]) ann->nodeval[i+1][ann->node_num_list[i+1] - 1] = ann->bias_flg_list[i+1];
    }
}

void ann_backward(ANN *ann, double *target, double rate)
{
    MATRIX prevo = { 1, 1 };
    int    i, j, k;

    if (!ann || ann->layer_num < 2 || !target) {
        printf("ann_backward: invalid ann or target !\n");
        return;
    }

    if (!ann->delta) ann->delta = matrix_create(1, ann->node_num_max);
    if (!ann->dtnew) ann->dtnew = matrix_create(1, ann->node_num_max);
    if (!ann->copy ) ann->copy  = matrix_create(ann->node_num_max, ann->node_num_max);
    if (!ann->dw   ) ann->dw    = matrix_create(ann->node_num_max, ann->node_num_max);

    for (i=ann->layer_num-2; i>=0; i--) {
        double *outa = ann->nodeval[i+1], totalerr;
        // calculate delta vector
        if (i == ann->layer_num-2) { // for output layer
            for (j=0; j<ann->node_num_list[i+1]; j++,outa++) {
                ann->delta->data[j] = -1 * (target[j] - *outa) * *outa * (1 - *outa);
            }
        } else { // for hidden layer
            for (j=0; j<ann->node_num_list[i+1]; j++,outa++) {
                for (totalerr=0,k=0; k<ann->copy->cols; k++) {
                    totalerr += ann->delta->data[k] * ann->copy->data[j*ann->copy->cols+k];
                }
                ann->dtnew->data[j] = totalerr * *outa * (1 - *outa);
            }
            memcpy(ann->delta->data, ann->dtnew->data, ann->node_num_list[i+1] * sizeof(double));
        }
        ann->delta->cols = ann->node_num_list[i+1];

        // make a copy of weight matrix
        if (i) {
            ann->copy->rows = ann->wmatrix[i].rows;
            ann->copy->cols = ann->wmatrix[i].cols;
            memcpy(ann->copy->data, ann->wmatrix[i].data, ann->copy->rows * ann->copy->cols * sizeof(double));
        }

        // calculate prev output vector
        prevo.rows = ann->node_num_list[i];
        prevo.data = ann->nodeval[i];

        ann->dw->rows = ann->node_num_list[i + 0];
        ann->dw->cols = ann->node_num_list[i + 1];
        matrix_multiply(ann->dw, &prevo, ann->delta);
        matrix_adjust  (&ann->wmatrix[i], ann->dw, rate);
    }
}

double ann_error(ANN *ann, double *target)
{
    double loss = 0;
    int    i;
    if (!ann) return 0;
    for (i=0; i<ann->node_num_list[ann->layer_num-1]; i++) {
        loss += 0.5 * pow(target[i] - ann->nodeval[ann->layer_num-1][i], 2);
    }
    return loss;
}

ANN* ann_load(char *file)
{
    ANN    *ann = NULL;
    FILE   *fp  = NULL;
    int     filesize, i;
    double *pdata;
    fp = fopen(file, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        filesize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        ann      = malloc(filesize);
        if (ann) {
            fread(ann, 1, filesize, fp);
            pdata = (double*)((uint8_t*)ann + sizeof(ANN));
            for (i=0; i<ann->layer_num; i++) {
                ann->nodeval[i] = pdata;
                pdata += ann->node_num_list[i];
            }
            for (i=0; i<ann->layer_num-1; i++) {
                ann->wmatrix[i].data = pdata;
                pdata += ann->wmatrix[i].rows * ann->wmatrix[i].cols;
            }
            ann->delta = ann->dtnew = ann->copy = ann->dw = NULL;
        } else {
            printf("ann_load: failed to allocate memory !\n");
        }
        fclose(fp);
    } else {
        printf("ann_load: failed to open file %s !\n", file);
    }
    return ann;
}

void ann_save(ANN *ann, char *file)
{
    FILE *fp = NULL;
    int   datasize, i;
    if (!ann || !file) {
        printf("ann_save: invalid samples or file !\n");
        return;
    }
    for (datasize=ann->node_num_list[0],i=1; i<ann->layer_num; i++) {
        datasize += ann->node_num_list[i];
        datasize += ann->node_num_list[i-1] * ann->node_num_list[i];
    }
    fp = fopen(file, "wb");
    if (fp) {
        fwrite(ann, 1, sizeof(ANN) + datasize * sizeof(double), fp);
        fclose(fp);
    } else {
        printf("ann_save: failed to open file %s !\n", file);
    }
}

void ann_dump(ANN *ann)
{
    int i, j;
    if (!ann) return;
    printf("\ndump ann info:\n");
    printf("- layer_num: %d\n", ann->layer_num);
    printf("- node_num_list: ");
    for (i=0; i<ann->layer_num; i++) {
        printf("%d ", ann->node_num_list[i]);
    }
    printf("\n");

    printf("- bias_flg_list: ");
    for (i=0; i<ann->layer_num; i++) {
        printf("%d ", ann->bias_flg_list[i]);
    }
    printf("\n\n");

    for (i=0; i<ann->layer_num; i++) {
        printf("- layer_%d: ", i);
        for (j=0; j<ann->node_num_list[i]; j++) {
            printf("%-8.5lf ", ann->nodeval[i][j]);
        }
        printf("\n\n");
    }

    for (i=0; i<ann->layer_num-1; i++) {
        printf("- matrix_%d:", i);
        matrix_print(&ann->wmatrix[i]);
        printf("\n");
    }
}

#if 1
#include "samples.h"
#include "bitmap.h"
int main(int argc, char *argv[])
{
    int node_num_list[ANN_MAX_LAYER] = {};
    int bias_flg_list[ANN_MAX_LAYER] = {};
    double rate, err_total, err_cur, err_max, err_min;
    uint32_t tick, sec, i, j;
    ANN     *ann;
    SAMPLES *samples;

    samples = samples_create(64, 11 * 22 + 1, 1);
    for (i=0; i<samples->num_samples; i++) {
        char name[256]; BMP mybmp;
        snprintf(name, sizeof(name), "pictures/asc_%03d.bmp", '0' + i);
        bmp_load(&mybmp, name);
        for (j=0; j<samples->num_input; j++) {
            samples_get_input(samples, i)[j] = bmp_getpixel(&mybmp, j);
        }
        samples_get_output(samples, i)[0] = i / (double)samples->num_samples;
        bmp_free(&mybmp);
    }

    if (argc < 2) {
        node_num_list[0] = samples->num_input;
        node_num_list[1] = 64 + 1; // 64 nodes + 1 bias
        node_num_list[2] = 64 + 1; // 64 nodes + 1 bias
        node_num_list[3] = samples->num_output;
        bias_flg_list[0] = 1;
        bias_flg_list[1] = 1;
        bias_flg_list[2] = 1;
        bias_flg_list[3] = 0;
        ann  = ann_create(4, node_num_list, bias_flg_list);
        tick = get_tick_count();
        sec  = 0;
        rate = 0.5;
        do {
            err_total = 0;
            err_max   = 0;
            err_min   = 999999999;
            for (j=0; j<samples->num_samples; j++) {
                ann_forward (ann, samples_get_input (samples, j));
                ann_backward(ann, samples_get_output(samples, j), rate);
                err_cur    = ann_error(ann, samples_get_output(samples, j));
                err_max    = MAX(err_max, err_cur);
                err_min    = MIN(err_min, err_cur);
                err_total += err_cur;
            }
            if (get_tick_count() - tick >= 1000) {
                printf("%5d. error_avg: %lf, error_total: %lf, error_max: %lf, error_min: %lf\n",
                       ++sec, err_total / samples->num_samples, err_total, err_max, err_min);
                fflush(stdout);
                tick += 1000;
            }
        } while (err_max > 0.00001);
        ann_save(ann, "ann.bin");
    } else {
        ann = ann_load(argv[1]);
    }

    printf("\n");
    for (i=0; i<samples->num_samples; i++) {
        ann_forward(ann, samples_get_input(samples, i));
        printf("output: %3d, %lf\n", (int)(ann_output(ann)[0] * samples->num_samples + 0.5), ann_output(ann)[0]);
    }

    ann_destroy(ann);
    samples_destroy(samples);
    return 0;
}
#endif
