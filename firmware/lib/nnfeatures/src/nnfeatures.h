/* nnfeatures: audio features for the neural-net costume.
 *
 * Single source of truth: this exact C file is compiled for the ESP32-S3
 * firmware AND loaded from Python (ml/features.py) to build the training set,
 * so training and inference cannot drift apart.
 *
 * Front end (shared by all modes): int16 PCM @16 kHz, always treated as exactly
 * 1 s (shorter is zero-padded, longer truncated) -> 61 Hann frames of 512, hop
 * 256 -> power spectrum -> 10 triangular mel bands (100-7600 Hz) -> per-frame
 * band energies E[frame][band].
 *
 * Every mode returns NNF_NUM_FEATURES (=10, the input layer) values:
 *
 *  NNF_MODE_MEAN      10 x mean over frames of ln(E + 1e-6). No timing. Baseline.
 *  NNF_MODE_CONTOURS  [0..4]  loudness in 5 time bins: mean of ln(sum_b E + 1e-6)
 *                     [5..9]  pitch in 5 time bins: energy-weighted mean band index
 *                             scaled 0..1 (0.5 if the bin is >20 dB below the loudest)
 *  NNF_MODE_HYBRID    [0..3]  loudness in 4 time bins (as above)
 *                     [4..6]  pitch in 3 time bins (as above)
 *                     [7..9]  3 orthonormal 2D-DCT coefficients of the 10x10 grid of
 *                             ln-mel (10 time bins x 10 bands), at (time,freq) =
 *                             (0,1) spectral tilt, (0,2) spectral bump, (1,1) tilt change
 *                             over time (low->high or high->low sweeps)
 *
 * Values are NOT normalised here; training computes mean/std and the firmware
 * applies them.
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
#define NNF_MAX_FRAMES    (1 + (NNF_CLIP_SAMPLES - NNF_FRAME_SIZE) / NNF_HOP_SIZE) /* 61 */

typedef enum {
    NNF_MODE_MEAN = 0,
    NNF_MODE_CONTOURS = 1,
    NNF_MODE_HYBRID = 2
} nnf_mode_t;

/* Extract NNF_NUM_FEATURES values from mono int16 PCM. Not thread-safe (lazy static tables). */
void nnf_extract(const int16_t *pcm, size_t n_samples, nnf_mode_t mode,
                 float out[NNF_NUM_FEATURES]);

#ifdef __cplusplus
}
#endif

#endif
