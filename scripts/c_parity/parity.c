/* The same algorithms as lib/linalg, in C (clang -O3 -ffp-contract=off), to ask one question: is the gap to numpy a gap of
 * cancho, or of any compiler without the vendor library's hand-tuned code? Same data (SplitMix64, docs/rng.md), same summation
 * order (docs/linalg.md section 2), so the result hash must equal linalg-check's; the time is the mean of REPS runs.
 *     cc -O3 -ffp-contract=off -o build/parity scripts/c_parity/parity.c
 *     build/parity KERNEL N D SEED REPS        (KERNEL dot axpy gemv_n gemv_t syrk chol)
 * Not part of the library; nothing here is shipped. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define K 256
static uint64_t mix(uint64_t z) { z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static const uint64_t GOLD = 0x9E3779B97F4A7C15ULL;
static uint64_t derive(uint64_t h, uint64_t c) { return mix((h ^ c) + GOLD); }
static double element(uint64_t seed, uint64_t i) { uint64_t z = mix(seed + (i + 1) * GOLD); return (double)(z >> 11) / 4503599627370496.0 - 1.0; }
static double *gen(long n, uint64_t seed) { double *a = aligned_alloc(64, ((n + 8) * 8 + 63) / 64 * 64); for (long i = 0; i < n; i++) a[i] = element(seed, i); return a; }
static uint64_t hash(const double *v, long n) { uint64_t h = 0; for (long i = 0; i < n; i++) { uint64_t b; memcpy(&b, v + i, 8); h += mix(b ^ mix(i)); } return h; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* the stack of lib/linalg/order.cho: slot s at st[s*w ..], levels lv[] */
static int settle(double *st, int *lv, int sp, long w) {
    lv[sp] = 0; sp++;
    while (sp >= 2 && lv[sp - 1] == lv[sp - 2]) {
        double *a = st + (long)(sp - 2) * w, *b = st + (long)(sp - 1) * w;
        for (long j = 0; j < w; j++) a[j] = a[j] + b[j];
        lv[sp - 2]++; sp--;
    }
    return sp;
}
static void finish(double *st, int sp, long w) {
    while (sp >= 2) { double *a = st + (long)(sp - 2) * w, *b = st + (long)(sp - 1) * w; for (long j = 0; j < w; j++) a[j] = a[j] + b[j]; sp--; }
}

static double dot(const double *x, const double *y, long n, double *st, int *lv) {
    if (n <= 0) return 0.0;
    long nb = (n - 1) / K + 1; int sp = 0; long b = 0;
    while (b < nb) {
        if (b + 8 <= nb && (b + 8) * K <= n) {
            double s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const double *xb = x + b * K, *yb = y + b * K;
            for (long j = 0; j < K; j++) {
                s[0] = s[0] + xb[j] * yb[j];            s[1] = s[1] + xb[K + j] * yb[K + j];
                s[2] = s[2] + xb[2 * K + j] * yb[2 * K + j]; s[3] = s[3] + xb[3 * K + j] * yb[3 * K + j];
                s[4] = s[4] + xb[4 * K + j] * yb[4 * K + j]; s[5] = s[5] + xb[5 * K + j] * yb[5 * K + j];
                s[6] = s[6] + xb[6 * K + j] * yb[6 * K + j]; s[7] = s[7] + xb[7 * K + j] * yb[7 * K + j];
            }
            for (int k = 0; k < 8; k++) { st[sp] = s[k]; sp = settle(st, lv, sp, 1); }
            b += 8;
        } else {
            long m = n - b * K; if (m > K) m = K; double s = 0.0;
            for (long j = 0; j < m; j++) s = s + x[b * K + j] * y[b * K + j];
            st[sp] = s; sp = settle(st, lv, sp, 1); b++;
        }
    }
    finish(st, sp, 1); return st[0];
}

static void axpy(double a, const double *x, double *y, long n) { for (long i = 0; i < n; i++) y[i] = y[i] + a * x[i]; }

static void gemv_n(const double *X, long n, long d, const double *w, double *y) {
    long i = 0;
    for (; i + 4 <= n; i += 4) {
        const double *r0 = X + i * d, *r1 = r0 + d, *r2 = r1 + d, *r3 = r2 + d;
        double z0 = 0, z1 = 0, z2 = 0, z3 = 0;
        for (long j = 0; j < d; j++) { double wj = w[j]; z0 = z0 + r0[j] * wj; z1 = z1 + r1[j] * wj; z2 = z2 + r2[j] * wj; z3 = z3 + r3[j] * wj; }
        y[i] = z0; y[i + 1] = z1; y[i + 2] = z2; y[i + 3] = z3;
    }
    for (; i < n; i++) { double z = 0; for (long j = 0; j < d; j++) z = z + X[i * d + j] * w[j]; y[i] = z; }
}

static void gemv_t(const double *X, long n, long d, const double *r, double *g, double *st, int *lv) {
    long nb = (n - 1) / K + 1; int sp = 0;
    for (long b = 0; b < nb; b++) {
        long lo = b * K, hi = lo + K; if (hi > n) hi = n;
        double *acc = st + (long)sp * d; for (long j = 0; j < d; j++) acc[j] = 0.0;
        long i = lo;
        for (; i + 4 <= hi; i += 4) {
            const double *a0 = X + i * d, *a1 = a0 + d, *a2 = a1 + d, *a3 = a2 + d; double c0 = r[i], c1 = r[i + 1], c2 = r[i + 2], c3 = r[i + 3];
            for (long j = 0; j < d; j++) { double s = acc[j]; s = s + a0[j] * c0; s = s + a1[j] * c1; s = s + a2[j] * c2; s = s + a3[j] * c3; acc[j] = s; }
        }
        for (; i < hi; i++) { const double *a0 = X + i * d; double c0 = r[i]; for (long j = 0; j < d; j++) acc[j] = acc[j] + a0[j] * c0; }
        sp = settle(st, lv, sp, d);
    }
    finish(st, sp, d); memcpy(g, st, d * 8);
}

static void syrk(const double *X, long n, long d, double *C, double *st, int *lv) {
    long nb = (n - 1) / K + 1; int sp = 0;
    for (long b = 0; b < nb; b++) {
        long lo = b * K, hi = lo + K; if (hi > n) hi = n;
        double *slot = st + (long)sp * d * d;
        for (long j0 = 0; j0 < d; j0 += 4) for (long k0 = j0; k0 < d; k0 += 4) {
            if (j0 + 4 <= d && k0 + 4 <= d) {
                double c[4][4] = {{0}};
                for (long i = lo; i < hi; i++) {
                    const double *p = X + i * d + j0, *q = X + i * d + k0;
                    for (int u = 0; u < 4; u++) for (int v = 0; v < 4; v++) c[u][v] = c[u][v] + p[u] * q[v];
                }
                for (int u = 0; u < 4; u++) for (int v = 0; v < 4; v++) slot[(j0 + u) * d + k0 + v] = c[u][v];
            } else {
                for (long j = j0; j < j0 + 4 && j < d; j++) for (long k = k0; k < k0 + 4 && k < d; k++) {
                    double c = 0; for (long i = lo; i < hi; i++) c = c + X[i * d + j] * X[i * d + k]; slot[j * d + k] = c;
                }
            }
        }
        sp = settle(st, lv, sp, d * d);
    }
    finish(st, sp, d * d); memcpy(C, st, d * d * 8);
}

static int chol(const double *A, double *L, long d) {  /* column by column, 8 rows at a time, ascending k */
    for (long j = 0; j < d; j++) {
        double s = A[j * d + j]; for (long k = 0; k < j; k++) s = s - L[j * d + k] * L[j * d + k];
        if (!(s > 0.0)) return 1;
        double ljj = sqrt(s); L[j * d + j] = ljj;
        long i = j + 1;
        for (; i + 8 <= d; i += 8) {
            double t[8]; for (int q = 0; q < 8; q++) t[q] = A[(i + q) * d + j];
            for (long k = 0; k < j; k++) { double c = L[j * d + k]; for (int q = 0; q < 8; q++) t[q] = t[q] - L[(i + q) * d + k] * c; }
            for (int q = 0; q < 8; q++) L[(i + q) * d + j] = t[q] / ljj;
        }
        for (; i < d; i++) { double t = A[i * d + j]; for (long k = 0; k < j; k++) t = t - L[i * d + k] * L[j * d + k]; L[i * d + j] = t / ljj; }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 6) return 2;
    const char *kn = argv[1]; long n = atol(argv[2]), d = atol(argv[3]); uint64_t seed = atol(argv[4]); long reps = atol(argv[5]);
    uint64_t s1 = derive(seed, 1), s2 = derive(seed, 2), s3 = derive(seed, 3);
    double *st = aligned_alloc(64, 40 * (d > 0 ? d * d : 1) * 8 + 4096); int lv[64]; double t0, t1; uint64_t h = 0; long cnt = 0;
    if (!strcmp(kn, "dot")) {
        double *x = gen(n, s1), *y = gen(n, s2); double r = 0; t0 = now(); for (long q = 0; q < reps; q++) r = dot(x, y, n, st, lv); t1 = now(); h = hash(&r, 1); cnt = 1;
    } else if (!strcmp(kn, "axpy")) {
        double *x = gen(n, s1), *y = gen(n, s2); double a = element(s3, 0); t0 = now(); for (long q = 0; q < reps; q++) axpy(a, x, y, n); t1 = now(); h = hash(y, n); cnt = n;
    } else if (!strcmp(kn, "gemv_n")) {
        double *X = gen(n * d, s1), *w = gen(d, s2), *y = calloc(n + 8, 8); t0 = now(); for (long q = 0; q < reps; q++) gemv_n(X, n, d, w, y); t1 = now(); h = hash(y, n); cnt = n;
    } else if (!strcmp(kn, "gemv_t")) {
        double *X = gen(n * d, s1), *r = gen(n, s2), *g = calloc(d + 8, 8); t0 = now(); for (long q = 0; q < reps; q++) gemv_t(X, n, d, r, g, st, lv); t1 = now(); h = hash(g, d); cnt = d;
    } else if (!strcmp(kn, "syrk")) {
        double *X = gen(n * d, s1), *C = calloc(d * d + 8, 8); double *full = calloc(d * d + 8, 8); t0 = now(); for (long q = 0; q < reps; q++) syrk(X, n, d, full, st, lv); t1 = now();
        for (long j = 0; j < d; j++) for (long k = j; k < d; k++) C[j * d + k] = full[j * d + k]; h = hash(C, d * d); cnt = d * d;
    } else if (!strcmp(kn, "chol")) {
        double *E = gen(n * n, s1), *A = calloc(n * n + 8, 8), *L = calloc(n * n + 8, 8);
        for (long i = 0; i < n; i++) for (long j = 0; j < n; j++) A[i * n + j] = (i == j) ? ((double)n + 1.0) + E[i * n + j] : 0.5 * (E[i * n + j] + E[j * n + i]);
        t0 = now(); for (long q = 0; q < reps; q++) chol(A, L, n); t1 = now(); h = hash(L, n * n); cnt = n * n;
    } else return 2;
    printf("hash %016llx\ncount %ld\nms %.6f\n", (unsigned long long)h, cnt, (t1 - t0) * 1000.0);
    return 0;
}
