#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "ffann.h"

int main(void)
{
    int   total_layers = 3;
    int   node_num_list[ANN_MAX_LAYER] = { 2, 3, 2 };
    int   activate_list[ANN_MAX_LAYER] = { 2, 2, 2 };
    float learn_rate = 0.125, target_error = 0.000001, total_error;
    float input_samples [4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    float output_samples[4][2] = { { 0, 0 }, { 1, 0 }, { 1, 0 }, { 0, 1 } };
    uint64_t total_times = 0;
    uint32_t tick, sec, i, j;

    ANN *ann = ann_create(total_layers, node_num_list, activate_list);
    tick = get_timestamp32_ms() + 1000;
    sec  = 0;

    do {
        total_error = 0;
        for (i = 0; i < 4; i++) {
            ann_forward (ann, input_samples [i]);
            ann_backward(ann, output_samples[i], learn_rate);
            total_error += ann_error(ann, output_samples[i]);
            total_times ++;
        }
        if (total_error < target_error || (int32_t)get_timestamp32_ms() - (int32_t)tick > 0) {
            tick += 1000;
            printf("%5ds, total_error: %f, total_times: %"PRIu64"\n", ++sec, total_error, total_times); fflush(stdout);
        }
    } while (total_error > target_error);

    printf("\n");
    for (i = 0; i < 4; i++) {
        float *outf;
        int    outn;
        ann_forward(ann, input_samples[i]);
        outf = ann_output(ann, &outn);
        printf("output_%d: ", i);
        for (j = 0; j < outn; j++) printf("%9f ", outf[j]);
        printf("\n");
    }
    printf("\n");

    ann_save(ann, "example1.bin");
    ann_dump(ann, NULL);
    ann_destroy(ann);
    return 0;
}
