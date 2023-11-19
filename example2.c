#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "ffann.h"
#include "samples.h"

int main(int argc, char *argv[])
{
    int total_layers = 4;
    int node_num_list[ANN_MAX_LAYER] = {};
    int activate_list[ANN_MAX_LAYER] = {};
    float learn_rate, batch_error_cur, batch_error_min, batch_error_max, batch_error_avg, batch_error_total, target_error;
    uint32_t tick, sec, ret, i, j, r, b;
    ANN     *ann;
    SAMPLES *samples;
    FILE    *fp1, *fp2;

    (void)ret;
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
            for (j = 0; j < 28 * 28; j++) {
                samples_get_input (samples, i)[j] = buf[j] / 255.01;
            }
            for (j = 0; j < 10; j++) {
                samples_get_output(samples, i)[j] = (out == j) / 1.01;
            }
        }
    }
    if (fp1) fclose(fp1);
    if (fp2) fclose(fp2);

    if (argc < 2) {
        node_num_list[0] = samples->num_input;
        node_num_list[1] = 28 * 28;
        node_num_list[2] = 14 * 14;
        node_num_list[3] = samples->num_output;
        activate_list[0] = 2;
        activate_list[1] = 2;
        activate_list[2] = 2;
        activate_list[3] = 2;
        ann  = ann_create(total_layers, node_num_list, activate_list);
        learn_rate   = 1 / 32.0;
        target_error = 0.01;

        tick = get_timestamp32_ms() + 1000;
        sec  = 0;
        r    = 0;
        do {
            r++;
            batch_error_min   = 1000000;
            batch_error_max   = 0;
            batch_error_total = 0;
            for (b = 0; b < 600; b++) {
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
                if (batch_error_max < target_error || (int32_t)get_timestamp32_ms() - (int32_t)tick > 0) {
                    printf("%5ds, round: %d, batch: %d, batch_err_cur: %f, batch_err_min: %f, batch_err_max: %f, batch_err_avg: %f\n",
                        ++sec, r, b + 1, batch_error_cur, batch_error_min, batch_error_max, batch_error_avg);
                    fflush(stdout);
                    tick += 1000;
                }
            }
            ann_save(ann, "example3.bin");
        } while (batch_error_max > target_error);
    } else {
        ann = ann_load(argv[1]);
    }

    printf("\n");
    for (i = 0; i < samples->num_samples; i++) {
        ann_forward(ann, samples_get_input(samples, i));
        printf("output_%d: ", i);
        for (j = 0; j < node_num_list[total_layers - 1]; j++) printf("%9f ", ann_output(ann)[j]);
        printf("error: %f", ann_error(ann, samples_get_output(samples, i)));
        printf("\n");
    }
    printf("\n");

    ann_destroy(ann);
    samples_destroy(samples);
    return 0;
}
