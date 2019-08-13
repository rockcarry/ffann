#include "ffann.h"

/*
the correct output
------------------

forward:
dump ann info:
- layer_num: 3
- node_num_list: 3 3 2
- bias_flg_list: 1 1 0

- layer_0: 0.05000  0.10000  1.00000

- layer_1: 0.59327  0.59688  1.00000

- layer_2: 0.75137  0.77293

- matrix_0:
0.15000  0.25000  0.00000
0.20000  0.30000  0.00000
0.35000  0.35000  0.00000

- matrix_1:
0.40000  0.50000
0.45000  0.55000
0.60000  0.60000

backward:
dump ann info:
- layer_num: 3
- node_num_list: 3 3 2
- bias_flg_list: 1 1 0

- layer_0: 0.05000  0.10000  1.00000

- layer_1: 0.59327  0.59688  1.00000

- layer_2: 0.75137  0.77293

- matrix_0:
0.14978  0.24975  0.00000
0.19956  0.29950  0.00000
0.34561  0.34502  0.00000

- matrix_1:
0.35892  0.51130
0.40867  0.56137
0.53075  0.61905
*/

int main(void)
{
    int node_num_list[ANN_MAX_LAYER] = { 3, 3, 2 };
    int bias_flg_list[ANN_MAX_LAYER] = { 1, 1, 0 };
    double input_samples [1][2] = { { 0.05, 0.10 } };
    double output_samples[1][2] = { { 0.01, 0.99 } };

    ANN *ann = ann_create(3, node_num_list, bias_flg_list);
    ann->wmatrix[0].data[0] = 0.15;
    ann->wmatrix[0].data[1] = 0.25;
    ann->wmatrix[0].data[2] = 0.00;
    ann->wmatrix[0].data[3] = 0.20;
    ann->wmatrix[0].data[4] = 0.30;
    ann->wmatrix[0].data[5] = 0.00;
    ann->wmatrix[0].data[6] = 0.35;
    ann->wmatrix[0].data[7] = 0.35;
    ann->wmatrix[0].data[8] = 0.00;

    ann->wmatrix[1].data[0] = 0.40;
    ann->wmatrix[1].data[1] = 0.50;
    ann->wmatrix[1].data[2] = 0.45;
    ann->wmatrix[1].data[3] = 0.55;
    ann->wmatrix[1].data[4] = 0.60;
    ann->wmatrix[1].data[5] = 0.60;

    ann_forward(ann, input_samples[0]);
    ann_dump(ann, NULL);
    ann_backward(ann, output_samples[0], 0.5);
    ann_dump(ann, NULL);
    ann_destroy(ann);
    return 0;
}


