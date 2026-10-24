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

/* Frames [start,end) of time bin i out of n_bins. */
static void bin_range(int i, int n_bins, int *start, int *end) {
    *start = i * NNF_MAX_FRAMES / n_bins;
    *end = (i + 1) * NNF_MAX_FRAMES / n_bins;
}

static void loudness_bins(float E[][NNF_NUM_FEATURES], int n_bins, float level, float *out) {
    for (int i = 0; i < n_bins; i++) {
        int s, e;
        bin_range(i, n_bins, &s, &e);
        float acc = 0.0f;
        for (int f = s; f < e; f++) {
            float tot = 0.0f;
            for (int b = 0; b < NNF_NUM_FEATURES; b++) tot += E[f][b];
            acc += logf(tot + 1e-6f);
        }
        out[i] = acc / (e - s) - level;
    }
}

static void pitch_bins(float E[][NNF_NUM_FEATURES], int n_bins, float *out) {
    float num[NNF_NUM_FEATURES], den[NNF_NUM_FEATURES], max_den = 0.0f;
    for (int i = 0; i < n_bins; i++) {
        int s, e;
        bin_range(i, n_bins, &s, &e);
        num[i] = den[i] = 0.0f;
        for (int f = s; f < e; f++)
            for (int b = 0; b < NNF_NUM_FEATURES; b++) {
                num[i] += b * E[f][b];
                den[i] += E[f][b];
            }
        if (den[i] > max_den) max_den = den[i];
    }
    for (int i = 0; i < n_bins; i++)
        out[i] = den[i] > 0.01f * max_den && den[i] > 0.0f
                     ? num[i] / (den[i] * (NNF_NUM_FEATURES - 1)) : 0.5f;
}

/* Orthonormal 2D DCT-II coefficient (kt, kf) of an N x N grid. */
static float dct2(float g[NNF_NUM_FEATURES][NNF_NUM_FEATURES], int kt, int kf) {
    const int n = NNF_NUM_FEATURES;
    float acc = 0.0f;
    for (int t = 0; t < n; t++)
        for (int f = 0; f < n; f++)
            acc += g[t][f] * cosf(PI_F * (2 * t + 1) * kt / (2 * n)) *
                             cosf(PI_F * (2 * f + 1) * kf / (2 * n));
    float at = kt == 0 ? sqrtf(1.0f / n) : sqrtf(2.0f / n);
    float af = kf == 0 ? sqrtf(1.0f / n) : sqrtf(2.0f / n);
    return at * af * acc;
}

void nnf_extract(const int16_t *pcm, size_t n_samples, int mode,
                 float out[NNF_NUM_FEATURES]) {
    if (!s_ready) init_tables();
    float re[NNF_FRAME_SIZE], im[NNF_FRAME_SIZE];
    float E[NNF_MAX_FRAMES][NNF_NUM_FEATURES];
    for (int fr = 0; fr < NNF_MAX_FRAMES; fr++) {
        size_t start = (size_t)fr * NNF_HOP_SIZE;
        for (int i = 0; i < NNF_FRAME_SIZE; i++) {
            size_t idx = start + i;
            re[i] = (idx < n_samples ? pcm[idx] / 32768.0f : 0.0f) * s_window[i];
            im[i] = 0.0f;
        }
        fft(re, im);
        for (int b = 0; b < NNF_NUM_FEATURES; b++) {
            float e = 0.0f;
            for (int k = 0; k < N_BINS; k++)
                if (s_fb[b][k] != 0.0f) e += s_fb[b][k] * (re[k] * re[k] + im[k] * im[k]);
            E[fr][b] = e;
        }
    }

    /* Level of the loudest frame; subtracted from ln-energy features if NNF_NORMALIZE. */
    float level = 0.0f;
    if (mode & NNF_NORMALIZE) {
        level = -1e30f;
        for (int fr = 0; fr < NNF_MAX_FRAMES; fr++) {
            float tot = 0.0f;
            for (int b = 0; b < NNF_NUM_FEATURES; b++) tot += E[fr][b];
            float l = logf(tot + 1e-6f);
            if (l > level) level = l;
        }
    }
    mode &= ~NNF_NORMALIZE;

    if (mode == NNF_MODE_CONTOURS) {
        loudness_bins(E, 5, level, out);
        pitch_bins(E, 5, out + 5);
    } else if (mode == NNF_MODE_HYBRID) {
        loudness_bins(E, 4, level, out);
        pitch_bins(E, 3, out + 4);
        float grid[NNF_NUM_FEATURES][NNF_NUM_FEATURES];
        for (int t = 0; t < NNF_NUM_FEATURES; t++) {
            int s, e;
            bin_range(t, NNF_NUM_FEATURES, &s, &e);
            for (int b = 0; b < NNF_NUM_FEATURES; b++) {
                float acc = 0.0f;
                for (int f = s; f < e; f++) acc += logf(E[f][b] + 1e-6f);
                grid[t][b] = acc / (e - s);
            }
        }
        out[7] = dct2(grid, 0, 1);
        out[8] = dct2(grid, 0, 2);
        out[9] = dct2(grid, 1, 1);
    } else {
        for (int b = 0; b < NNF_NUM_FEATURES; b++) {
            float acc = 0.0f;
            for (int fr = 0; fr < NNF_MAX_FRAMES; fr++) acc += logf(E[fr][b] + 1e-6f);
            out[b] = acc / NNF_MAX_FRAMES - level;
        }
    }
}
