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
    float batch_loss_cur, batch_loss_min, batch_loss_max, batch_loss_avg, batch_loss_total;
    float learn_rate  = 1 / 32.0;
    float target_loss = 0.01;
    char *model_file  = "example2.bin";
    int   run_type    = 0; // 0 - inference, 1 - create train, 2 - load train
    uint32_t tick, sec, ret = -1, i, j, r, b;
    ANN     *ann     = NULL;
    SAMPLES *samples = NULL;
    FILE    *fp1, *fp2;

    for (i = 1; i < argc; i++) {
        if      (strstr(argv[i], "--learn_rate=" )) learn_rate  = atof(argv[i] + sizeof("--learn_rate=" ) - 1);
        else if (strstr(argv[i], "--target_loss=")) target_loss = atof(argv[i] + sizeof("--target_loss=") - 1);
        else if (strstr(argv[i], "--infer="      )) { run_type = 0, model_file = argv[i] + sizeof("--infer=" ) - 1; }
        else if (strstr(argv[i], "--create="     )) { run_type = 1, model_file = argv[i] + sizeof("--create=") - 1; }
        else if (strstr(argv[i], "--load="       )) { run_type = 2, model_file = argv[i] + sizeof("--load="  ) - 1; }
    }

    printf("learn_rate : %f\n", learn_rate );
    printf("target_loss: %f\n", target_loss);
    printf("model_file : %s\n", model_file );
    printf("\n");

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
            for (j = 0; j < 10;      j++) samples_get_output(samples, i)[j] = (out == j) ? 1 : 0;
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
        node_num_list[2] = 20;
        node_num_list[3] = samples->num_output;
        activate_list[0] = 2;
        activate_list[1] = 2;
        activate_list[2] = 2;
        activate_list[3] = 3;
        ann = ann_create(total_layers, node_num_list, activate_list);
        break;
    }
    if (!ann) {
        printf("failed to load/create ffann model file !\n");
        goto done;
    }

    if (run_type == 0) {
        int correct = 0;
        printf("\n");
        for (i = samples->num_samples - 100; i < samples->num_samples; i++) {
            float *outf, *outt, maxf, maxt, score;
            int    outn, clsf = 0, clst = 0;
            ann_forward(ann, samples_get_input(samples, i));
            outf = ann_output(ann, &outn);
            outt = samples_get_output(samples, i);
            printf("output_%d: ", i);
            for (maxf = 0, maxt = 0, j = 0; j < outn; j++) {
                printf("%8.5f ", outf[j]);
                if (maxf < outf[j]) { maxf = outf[j]; clsf = j; }
                if (maxt < outt[j]) { maxt = outt[j]; clst = j; }
            }
            correct += clsf == clst;
            printf("%c ", clsf == clst ? ' ' : 'F');
            score = 1 - fabsf(maxf - 1);
            printf("loss: %8.5f, class: %d, score: %.2f", ann_loss(ann, samples_get_output(samples, i)), clsf, score);
            printf("\n");
        }
        printf("precision: %5.2f %%\n", 100.0 * (i - correct) / i);
    } else {
        tick = get_timestamp32_ms() + 1000;
        sec  = 0;
        r    = 0;
        do {
            r++;
            batch_loss_min   = 1000000;
            batch_loss_max   = 0;
            batch_loss_total = 0;
            for (b = 0; b < 600 - 1; b++) {
                batch_loss_cur = 0;
                for (i = 0; i < 100; i++) {
                    ann_forward (ann, samples_get_input (samples, b * 100 + i));
                    ann_backward(ann, samples_get_output(samples, b * 100 + i), learn_rate);
                    batch_loss_cur += ann_loss(ann, samples_get_output(samples, b * 100 + i));
                }
                batch_loss_min    = MIN(batch_loss_min, batch_loss_cur);
                batch_loss_max    = MAX(batch_loss_max, batch_loss_cur);
                batch_loss_total += batch_loss_cur;
                batch_loss_avg    = batch_loss_total / (b + 1);
                if (fpclassify(batch_loss_cur) != FP_NORMAL) { printf("got float point abnormal, model training failed !\n"); goto done; }
                if (batch_loss_avg < target_loss || (int32_t)get_timestamp32_ms() - (int32_t)tick > 0) {
                    printf("%5ds, round: %d, batch: %d, batch_loss_cur: %f, batch_loss_min: %f, batch_loss_max: %f, batch_loss_avg: %f\n",
                        ++sec, r, b + 1, batch_loss_cur, batch_loss_min, batch_loss_max, batch_loss_avg);
                    fflush(stdout);
                    tick += 1000;
                }
            }
            ann_save(ann, model_file);
        } while (batch_loss_avg > target_loss);
        printf("model training finish !!\n");
    }
    ret = 0;

done:
    ann_destroy(ann);
    samples_destroy(samples);
    return ret;
}
