#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "ffann.h"

int main(void)
{
    int node_num_list[ANN_MAX_LAYER] = { 2, 4, 1 };
    float learn_rate = 0.5, total_error;
    float input_samples [4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    float output_samples[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };
    uint32_t tick, sec, i;
    ANN *ann = NULL;

    ann = ann_create(3, node_num_list);
    tick= get_timestamp32_ms();
    sec = 0;

    do {
        total_error = 0;
        for (i = 0; i < 4; i++) {
            ann_forward (ann, input_samples [i]);
            ann_backward(ann, output_samples[i], learn_rate);
            total_error += ann_error(ann, output_samples[i]);
        }
        if (get_timestamp32_ms() > tick + 1000) {
            tick += 1000;
            printf("%5ds, total_error: %f\n", ++sec, total_error);
            fflush(stdout);
        }
    } while (total_error > 0.000002);

    printf("\n");
    for (i = 0; i < 4; i++) {
        ann_forward(ann, input_samples[i]);
        printf("output_%d: %lf\n", i, ann_output(ann)[0]);
    }
    printf("\n");

    ann_save(ann, "example1.bin");
    ann_dump(ann, NULL);
    ann_destroy(ann);
    return 0;
}

