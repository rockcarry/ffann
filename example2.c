#include "ffann.h"
#include "samples.h"
#include "bitmap.h"

int main(int argc, char *argv[])
{
    int node_num_list[ANN_MAX_LAYER] = {};
    int bias_flg_list[ANN_MAX_LAYER] = {};
    double learn_rate, err_total, err_cur, err_max, err_min;
    uint32_t tick, sec, i, j;
    ANN     *ann;
    SAMPLES *samples;

    samples = samples_create(64, 11 * 22, 1);
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
        learn_rate = 0.5;
        do {
            err_total = 0;
            err_max   = 0;
            err_min   = 999999999;
            for (j=0; j<samples->num_samples; j++) {
                ann_forward (ann, samples_get_input (samples, j));
                ann_backward(ann, samples_get_output(samples, j), learn_rate);
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
        ann_save(ann, "example2.bin");
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

