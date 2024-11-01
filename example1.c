#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "ffann.h"

#define SAMPLE_NORMALIZE

int main(void)
{
    int   total_layers = 3;
    int   node_num_list[ANN_MAX_LAYER] = { 2, 4, 2 };
    int   activate_list[ANN_MAX_LAYER] = { 2, 2, 2 };
    float learn_rate = 0.1, target_loss = 0.00001, total_loss;
    float input_samples [4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    float output_samples[4][2] = { { 0, 0 }, { 1, 0 }, { 1, 0 }, { 0, 1 } };
    uint64_t total_times = 0;
    uint32_t tick, sec, i, j;

    ANN *ann = ann_create(total_layers, node_num_list, activate_list);
    tick = get_timestamp32_ms() + 1000;
    sec  = 0;

#ifdef SAMPLE_NORMALIZE
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 2; j++) {
            input_samples [i][j] = input_samples[i][j] > 0.5 ? 0.999 : 0.001;
            output_samples[i][j] = input_samples[i][j] > 0.5 ? 0.999 : 0.001;
        }
    }
#endif

    do {
        total_loss = 0;
        for (i = 0; i < 4; i++) {
            ann_forward (ann, input_samples [i]);
            ann_backward(ann, output_samples[i], learn_rate);
            total_loss += ann_loss(ann, output_samples[i]);
            total_times++;
        }
        if (total_loss < target_loss || (int32_t)get_timestamp32_ms() - (int32_t)tick > 0) {
            tick += 1000;
            printf("%5ds, total_loss: %f, total_times: %"PRIu64"\n", ++sec, total_loss, total_times); fflush(stdout);
        }
    } while (total_loss > target_loss);

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
