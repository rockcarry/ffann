#include "ffann.h"

int main(void)
{
    int node_num_list[ANN_MAX_LAYER] = { 3, 4, 1 };
    int bias_flg_list[ANN_MAX_LAYER] = { 1, 1, 0 };
    double learn_rate = 0.5, total_error;
    double input_samples [4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    double output_samples[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };
    uint32_t tick, sec, i;
    ANN *ann = NULL;

    ann = ann_create(3, node_num_list, bias_flg_list);
    tick= get_tick_count();
    sec = 0;

    do {
        total_error = 0;
        for (i=0; i<4; i++) {
            ann_forward (ann, input_samples [i]);
            ann_backward(ann, output_samples[i], learn_rate);
            total_error += ann_error(ann, output_samples[i]);
        }
        if (get_tick_count() > tick + 1000) {
            tick += 1000;
            printf("%5ds, total_error: %.7lf\n", ++sec, total_error);
            fflush(stdout);
        }
    } while (total_error > 0.000001);

    printf("\n");
    for (i=0; i<4; i++) {
        ann_forward(ann, input_samples[i]);
        printf("output_%d: %lf\n", i, ann_output(ann)[0]);
    }

    ann_save(ann, "example1.bin");
    ann_dump(ann, NULL);
    ann_destroy(ann);
    return 0;
}


