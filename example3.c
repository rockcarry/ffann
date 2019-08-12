#include "ffann.h"
#include "samples.h"

int main(int argc, char *argv[])
{
    int node_num_list[ANN_MAX_LAYER] = {};
    int bias_flg_list[ANN_MAX_LAYER] = {};
    double learn_rate, batch_error, target_error;
    uint32_t i, j, r;
    ANN     *ann;
    SAMPLES *samples;
    FILE    *fp1, *fp2;

    samples = samples_create(60*1000, 28 * 28 + 1, 10);
    fp1     = fopen("mnist/train-images.idx3-ubyte", "rb");
    fp2     = fopen("mnist/train-labels.idx1-ubyte", "rb");
    if (fp1 && fp2) {
        fseek(fp1, 16, SEEK_SET);
        fseek(fp2, 8 , SEEK_SET);
        for (i=0; i<samples->num_samples; i++) {
            uint8_t buf[28 * 28], out;
            fread( buf, 28 * 28, 1, fp1);
            fread(&out, 1, 1, fp2);
            for (j=0; j<28*28; j++) {
                samples_get_input(samples, i)[j] = (buf[j] - 128) / 128;
            }
            for (j=0; j<10; j++) {
                samples_get_output(samples, i)[j] = !!(out == j);
            }
        }
    }
    if (fp1) fclose(fp1);
    if (fp2) fclose(fp2);

    if (argc < 2) {
        node_num_list[0] = samples->num_input;
        node_num_list[1] = 32 + 1; // 32 nodes + 1 bias
        node_num_list[2] = 16 + 1; // 16 nodes + 1 bias
        node_num_list[3] = samples->num_output;
        bias_flg_list[0] = 1;
        bias_flg_list[1] = 1;
        bias_flg_list[2] = 1;
        bias_flg_list[3] = 0;
        ann  = ann_create(4, node_num_list, bias_flg_list);
        learn_rate   = 0.2;
        target_error = 100;
        for (r=0; r<1000; r++) {
            for (i=0; i<samples->num_samples/1000; i++) {
                do {
                    batch_error = 0;
                    for (j=0; j<1000; j++) {
                        ann_forward (ann, samples_get_input (samples, i * 1000 + j));
                        ann_backward(ann, samples_get_output(samples, i * 1000 + j), learn_rate);
                        batch_error += ann_error(ann, samples_get_output(samples, i * 1000 + j));
                    }
                    printf("round: %d, batch: %d, learn_rate: %lf, target_error: %lf, batch_error: %lf\n", r, i, learn_rate, target_error, batch_error);
                    fflush(stdout);
                } while (batch_error > target_error);
            }
            learn_rate   *= 0.8;
            target_error *= 0.8;
        }
        ann_save(ann, "example3.bin");
    } else {
        ann = ann_load(argv[1]);
    }

    ann_destroy(ann);
    samples_destroy(samples);
    return 0;
}

