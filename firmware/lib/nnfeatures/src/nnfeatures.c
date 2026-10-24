#include "nnfeatures.h"

#include <math.h>
#include <string.h>

#define N_BINS (NNF_FRAME_SIZE / 2 + 1)
#define PI_F 3.14159265358979323846f

static int s_ready;
static float s_window[NNF_FRAME_SIZE];
static float s_fb[NNF_NUM_FEATURES][N_BINS];
static float s_cos[NNF_FRAME_SIZE / 2], s_sin[NNF_FRAME_SIZE / 2];

static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

static void init_tables(void) {
    for (int i = 0; i < NNF_FRAME_SIZE; i++)  /* periodic Hann */
        s_window[i] = 0.5f - 0.5f * cosf(2.0f * PI_F * i / NNF_FRAME_SIZE);
    for (int i = 0; i < NNF_FRAME_SIZE / 2; i++) {
        s_cos[i] = cosf(2.0f * PI_F * i / NNF_FRAME_SIZE);
        s_sin[i] = -sinf(2.0f * PI_F * i / NNF_FRAME_SIZE);
    }
    /* NNF_NUM_FEATURES triangles need NUM+2 edge points, evenly spaced in mel. */
    float edges[NNF_NUM_FEATURES + 2];
    float m0 = hz_to_mel(NNF_FMIN_HZ), m1 = hz_to_mel(NNF_FMAX_HZ);
    for (int i = 0; i < NNF_NUM_FEATURES + 2; i++)
        edges[i] = mel_to_hz(m0 + (m1 - m0) * i / (NNF_NUM_FEATURES + 1));
    for (int b = 0; b < NNF_NUM_FEATURES; b++) {
        for (int k = 0; k < N_BINS; k++) {
            float hz = (float)k * NNF_SAMPLE_RATE / NNF_FRAME_SIZE;
            float lo = edges[b], mid = edges[b + 1], hi = edges[b + 2], w = 0.0f;
            if (hz > lo && hz <= mid) w = (hz - lo) / (mid - lo);
            else if (hz > mid && hz < hi) w = (hi - hz) / (hi - mid);
            s_fb[b][k] = w;
        }
    }
    s_ready = 1;
}

/* In-place iterative radix-2 complex FFT of size NNF_FRAME_SIZE. */
static void fft(float *re, float *im) {
    const int n = NNF_FRAME_SIZE;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; k++) {
                float wr = s_cos[k * step], wi = s_sin[k * step];
                int a = i + k, b = a + half;
                float tr = re[b] * wr - im[b] * wi;
                float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
        }
    }
}

void nnf_extract(const int16_t *pcm, size_t n_samples, float out[NNF_NUM_FEATURES]) {
    if (!s_ready) init_tables();
    float re[NNF_FRAME_SIZE], im[NNF_FRAME_SIZE], acc[NNF_NUM_FEATURES] = {0};
    size_t frames = 0;
    for (size_t start = 0; start + NNF_FRAME_SIZE <= n_samples; start += NNF_HOP_SIZE, frames++) {
        for (int i = 0; i < NNF_FRAME_SIZE; i++) {
            re[i] = (pcm[start + i] / 32768.0f) * s_window[i];
            im[i] = 0.0f;
        }
        fft(re, im);
        for (int b = 0; b < NNF_NUM_FEATURES; b++) {
            float e = 0.0f;
            for (int k = 0; k < N_BINS; k++)
                if (s_fb[b][k] != 0.0f) e += s_fb[b][k] * (re[k] * re[k] + im[k] * im[k]);
            acc[b] += logf(e + 1e-6f);
        }
    }
    for (int b = 0; b < NNF_NUM_FEATURES; b++) out[b] = frames ? acc[b] / frames : 0.0f;
}
