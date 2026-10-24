/* nnforward: the costume's 10-10-10-10 network, forward pass only.
 *
 * Like nnfeatures, this one C source runs on the ESP32 AND is loaded from Python
 * (ml/nn.py) so evaluation measures exactly what the costume will do.
 *
 * Layers (4 x 10 neurons, 3 weight matrices):
 *   act[0]  input     the 10 features, z-scored; for display, squashed to 0..1
 *   act[1]  hidden 1  sigmoid(W[0] z + b[0])
 *   act[2]  hidden 2  sigmoid(W[1] a1 + b[1])
 *   act[3]  output    softmax(W[2] a2 + b[2])     (index = label order in tools/words.py)
 * Every act[][] value is in 0..1, ready to use as LED brightness.
 */
#ifndef NNFORWARD_H
#define NNFORWARD_H

#ifdef __cplusplus
extern "C" {
#endif

#define NN_N       10   /* neurons per layer */
#define NN_LAYERS  4    /* including the input layer */

typedef struct {
    float mean[NN_N];                 /* feature normalisation (from training set) */
    float std[NN_N];
    float W[NN_LAYERS - 1][NN_N][NN_N]; /* W[layer][out][in] */
    float b[NN_LAYERS - 1][NN_N];
} nn_weights_t;

/* features: raw nnfeatures output. Fills act[4][10]; returns the winning output index. */
int nn_forward(const nn_weights_t *w, const float features[NN_N], float act[NN_LAYERS][NN_N]);

#ifdef __cplusplus
}
#endif

#endif
