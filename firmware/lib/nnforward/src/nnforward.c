#include "nnforward.h"

#include <math.h>

/* Display value of the input neuron: z of -2..+2 maps to 0..1. */
static float display_input(float z) {
    float v = 0.5f + z * 0.25f;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

int nn_forward(const nn_weights_t *w, const float features[NN_N], float act[NN_LAYERS][NN_N]) {
    float z[NN_N], a[NN_N], next[NN_N];
    for (int i = 0; i < NN_N; i++) {
        z[i] = (features[i] - w->mean[i]) / w->std[i];
        act[0][i] = display_input(z[i]);
        a[i] = z[i];  /* the network sees the unclamped z */
    }
    for (int l = 0; l < NN_LAYERS - 1; l++) {
        for (int o = 0; o < NN_N; o++) {
            float s = w->b[l][o];
            for (int i = 0; i < NN_N; i++) s += w->W[l][o][i] * a[i];
            next[o] = s;
        }
        if (l < NN_LAYERS - 2) {
            for (int o = 0; o < NN_N; o++) a[o] = act[l + 1][o] = sigmoid(next[o]);
        } else {  /* softmax, max subtracted for stability */
            float mx = next[0], sum = 0.0f;
            for (int o = 1; o < NN_N; o++) if (next[o] > mx) mx = next[o];
            for (int o = 0; o < NN_N; o++) { next[o] = expf(next[o] - mx); sum += next[o]; }
            for (int o = 0; o < NN_N; o++) act[l + 1][o] = next[o] / sum;
        }
    }
    int best = 0;
    for (int o = 1; o < NN_N; o++) if (act[NN_LAYERS - 1][o] > act[NN_LAYERS - 1][best]) best = o;
    return best;
}
