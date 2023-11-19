#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "ffann.h"
#include "samples.h"

int main(int argc, char *argv[])
{
    int   total_layers = 4;
    int   node_num_list[ANN_MAX_LAYER] = {};
    int   activate_list[ANN_MAX_LAYER] = {};
    float batch_error_cur, batch_error_min, batch_error_max, batch_error_avg, batch_error_total;
    float learn_rate   = 1 / 16.0;
    float target_error = 0.1;
    char *model_file   = "example2.bin";
    int   run_type     = 0; // 0 - inference, 1 - create train, 2 - load train
    uint32_t tick, sec, ret = -1, i, j, r, b;
    ANN     *ann     = NULL;
    SAMPLES *samples = NULL;;
    FILE    *fp1, *fp2;

    for (i = 1; i < argc; i++) {
        if      (strstr(argv[i], "--learn_rate=")) learn_rate   = atof(argv[i] + sizeof("--learn_rate=") - 1);
        else if (strstr(argv[i], "--target_err=")) target_error = atof(argv[i] + sizeof("--target_err=") - 1);
        else if (strstr(argv[i], "--infer="     )) { run_type = 0, model_file = argv[i] + sizeof("--infer=" ) - 1; }
        else if (strstr(argv[i], "--create="    )) { run_type = 1, model_file = argv[i] + sizeof("--create=") - 1; }
        else if (strstr(argv[i], "--load="      )) { run_type = 2, model_file = argv[i] + sizeof("--load="  ) - 1; }
    }

    printf("learn_rate: %f\n", learn_rate  );
    printf("target_err: %f\n", target_error);
    printf("model_file: %s\n", model_file  );

    // load samples
    samples = samples_create(60*1000, 28 * 28, 10);
    fp1     = fopen("mnist/train-images.idx3-ubyte", "rb");
    fp2     = fopen("mnist/train-labels.idx1-ubyte", "rb");
    if (fp1 && fp2) {
        fseek(fp1, 16, SEEK_SET);
        fseek(fp2, 8 , SEEK_SET);
        for (i = 0; i < samples->num_samples; i++) {
            uint8_t buf[28 * 28], out;
            ret = fread( buf, 28 * 28, 1, fp1);
            ret = fread(&out, 1, 1, fp2);
            for (j = 0; j < 28 * 28; j++) samples_get_input (samples, i)[j] = buf[j] ? buf[j] / 255.01 : 0.0001;
            for (j = 0; j < 10;      j++) samples_get_output(samples, i)[j] = (out == j) ? 0.9999 : 0.0001;
        }
    }
    if (fp1) fclose(fp1);
    if (fp2) fclose(fp2);
    if (!fp1 || !fp2) {
        printf("failed to load minist samples !\n");
        goto done;
    }

    // load/create model file
    switch (run_type) {
    case 0: case 2:
        ann = ann_load(model_file);
        break;
    case 1:
        node_num_list[0] = samples->num_input;
        node_num_list[1] = 64;
        node_num_list[2] = 16;
        node_num_list[3] = samples->num_output;
        activate_list[0] = 2;
        activate_list[1] = 2;
        activate_list[2] = 2;
        activate_list[3] = 2;
        ann = ann_create(total_layers, node_num_list, activate_list);
        break;
    }
    if (!ann) {
        printf("failed to load/create ffann model file !\n");
        goto done;
    }

    if (run_type == 0) {
        printf("\n");
        for (i = samples->num_samples - 100; i < samples->num_samples; i++) {
            float *outf, max, score;
            int    outn, cls = 0;
            ann_forward(ann, samples_get_input(samples, i));
            outf = ann_output(ann, &outn);
            printf("output_%d: ", i);
            for (max = 0, j = 0; j < outn; j++) {
                printf("%9f ", outf[j]);
                if (max < outf[j]) {
                    max = outf[j];
                    cls = j;
                }
            }
            score = 1 - fabsf(max - 1);
            printf("error: %f, class: %d, score: %.2f", ann_error(ann, samples_get_output(samples, i)), cls, score);
            printf("\n");
        }
        printf("\n");
    } else {
        tick = get_timestamp32_ms() + 1000;
        sec  = 0;
        r    = 0;
        do {
            r++;
            batch_error_min   = 1000000;
            batch_error_max   = 0;
            batch_error_total = 0;
            for (b = 0; b < 600 - 1; b++) {
                batch_error_cur = 0;
                for (i = 0; i < 100; i++) {
                    ann_forward (ann, samples_get_input (samples, b * 100 + i));
                    ann_backward(ann, samples_get_output(samples, b * 100 + i), learn_rate);
                    batch_error_cur += ann_error(ann, samples_get_output(samples, b * 100 + i));
                }
                batch_error_min    = MIN(batch_error_min, batch_error_cur);
                batch_error_max    = MAX(batch_error_max, batch_error_cur);
                batch_error_total += batch_error_cur;
                batch_error_avg    = batch_error_total / (b + 1);
                if (fpclassify(batch_error_cur) != FP_NORMAL) { printf("got float point abnormal, model training failed !\n"); goto done; }
                if (batch_error_max < target_error || (int32_t)get_timestamp32_ms() - (int32_t)tick > 0) {
                    printf("%5ds, round: %d, batch: %d, batch_err_cur: %f, batch_err_min: %f, batch_err_max: %f, batch_err_avg: %f\n",
                        ++sec, r, b + 1, batch_error_cur, batch_error_min, batch_error_max, batch_error_avg);
                    fflush(stdout);
                    tick += 1000;
                }
            }
            ann_save(ann, model_file);
        } while (batch_error_max > target_error);
        printf("model training finish !!\n");
    }
    ret = 0;

done:
    ann_destroy(ann);
    samples_destroy(samples);
    return ret;
}
