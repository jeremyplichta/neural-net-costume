/* nnfeatures: audio features for the neural-net costume.
 *
 * Single source of truth: this exact C file is compiled for the ESP32-S3
 * firmware AND loaded from Python (ml/features.py) to build the training set,
 * so training and inference cannot drift apart.
 *
 * Pipeline: int16 PCM @16 kHz -> 512-sample Hann frames, hop 256 -> power
 * spectrum -> NNF_NUM_FEATURES triangular mel bands (100-7600 Hz) ->
 * ln(energy + 1e-6) -> mean over frames.
 */
#ifndef NNFEATURES_H
#define NNFEATURES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NNF_SAMPLE_RATE   16000
#define NNF_CLIP_SAMPLES  16000   /* 1 s */
#define NNF_FRAME_SIZE    512
#define NNF_HOP_SIZE      256
#define NNF_NUM_FEATURES  10      /* = neurons in the input layer */
#define NNF_FMIN_HZ       100.0f
#define NNF_FMAX_HZ       7600.0f

/* Extract NNF_NUM_FEATURES values from n_samples of mono int16 PCM.
 * n_samples must be >= NNF_FRAME_SIZE. Not thread-safe (lazy static tables). */
void nnf_extract(const int16_t *pcm, size_t n_samples, float out[NNF_NUM_FEATURES]);

#ifdef __cplusplus
}
#endif

#endif
