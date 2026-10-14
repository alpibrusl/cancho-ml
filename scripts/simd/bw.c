/* Single-core read-bandwidth probe (the ceiling for the memory-bound kernels): sums an array of N doubles with NEON, 8 vector
 * accumulators, three load styles: plain 16-byte loads, vld1q_f64_x4 (64-byte loads), and x4 with software prefetch 1 KB ahead.
 *     cc -O3 -mcpu=native -o build/bw scripts/simd/bw.c ; build/bw N    (min of 15 runs, GB/s = 8 N / time) */
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double s_plain(const double *x, long n) { float64x2_t a[8]; for (int r = 0; r < 8; r++) a[r] = vdupq_n_f64(0);
    for (long i = 0; i + 16 <= n; i += 16) for (int r = 0; r < 8; r++) a[r] = vaddq_f64(a[r], vld1q_f64(x + i + 2 * r));
    for (int r = 1; r < 8; r++) a[0] = vaddq_f64(a[0], a[r]); return vaddvq_f64(a[0]); }
static double s_x4(const double *x, long n) { float64x2_t a[8]; for (int r = 0; r < 8; r++) a[r] = vdupq_n_f64(0);
    for (long i = 0; i + 16 <= n; i += 16) { float64x2x4_t u = vld1q_f64_x4(x + i), v = vld1q_f64_x4(x + i + 8);
        a[0] = vaddq_f64(a[0], u.val[0]); a[1] = vaddq_f64(a[1], u.val[1]); a[2] = vaddq_f64(a[2], u.val[2]); a[3] = vaddq_f64(a[3], u.val[3]);
        a[4] = vaddq_f64(a[4], v.val[0]); a[5] = vaddq_f64(a[5], v.val[1]); a[6] = vaddq_f64(a[6], v.val[2]); a[7] = vaddq_f64(a[7], v.val[3]); }
    for (int r = 1; r < 8; r++) a[0] = vaddq_f64(a[0], a[r]); return vaddvq_f64(a[0]); }
static double s_pf(const double *x, long n) { float64x2_t a[8]; for (int r = 0; r < 8; r++) a[r] = vdupq_n_f64(0);
    for (long i = 0; i + 16 <= n; i += 16) { __builtin_prefetch(x + i + 128); __builtin_prefetch(x + i + 136); float64x2x4_t u = vld1q_f64_x4(x + i), v = vld1q_f64_x4(x + i + 8);
        a[0] = vaddq_f64(a[0], u.val[0]); a[1] = vaddq_f64(a[1], u.val[1]); a[2] = vaddq_f64(a[2], u.val[2]); a[3] = vaddq_f64(a[3], u.val[3]);
        a[4] = vaddq_f64(a[4], v.val[0]); a[5] = vaddq_f64(a[5], v.val[1]); a[6] = vaddq_f64(a[6], v.val[2]); a[7] = vaddq_f64(a[7], v.val[3]); }
    for (int r = 1; r < 8; r++) a[0] = vaddq_f64(a[0], a[r]); return vaddvq_f64(a[0]); }
static double s_streams(const double *x, long n, int S) {  /* S concurrent sequential streams (n/S apart), 4 vector loads per stream per step */
    float64x2_t a[8]; for (int r = 0; r < 8; r++) a[r] = vdupq_n_f64(0); long q = n / S / 8 * 8;
    for (long i = 0; i + 8 <= q; i += 8) for (int t = 0; t < S; t++) { const double *p = x + t * q + i; for (int r = 0; r < 4; r++) a[(t * 4 + r) & 7] = vaddq_f64(a[(t * 4 + r) & 7], vld1q_f64(p + 2 * r)); }
    for (int r = 1; r < 8; r++) a[0] = vaddq_f64(a[0], a[r]); return vaddvq_f64(a[0]); }
static double s_s2(const double *x, long n) { return s_streams(x, n, 2); }
static double s_s4(const double *x, long n) { return s_streams(x, n, 4); }
static double s_s8(const double *x, long n) { return s_streams(x, n, 8); }
int main(int argc, char **argv) {
    long n = atol(argv[1]); double *x = aligned_alloc(64, (n * 8 + 63) / 64 * 64); for (long i = 0; i < n; i++) x[i] = 1.0 + (i & 7);
    double (*f[6])(const double *, long) = {s_plain, s_x4, s_pf, s_s2, s_s4, s_s8}; const char *nm[6] = {"plain 16B loads", "x4 64B loads", "x4 + prefetch", "2 streams", "4 streams", "8 streams"}; volatile double sink = 0;
    for (int k = 0; k < 6; k++) { double best = 1e9; for (int rep = 0; rep < 15; rep++) { double t = now(); sink += f[k](x, n); t = now() - t; if (t < best) best = t; }
        printf("N=%ld (%.1f MB)  %-16s %9.1f us  %6.1f GB/s\n", n, n * 8e-6, nm[k], best * 1e6, n * 8 / best / 1e9); }
    return 0;
}
