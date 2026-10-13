/* The same algorithms as lib/linalg, in C (clang -O3 -ffp-contract=off), to ask one question: is the gap to numpy a gap of
 * cancho, or of any compiler without the vendor library's hand-tuned code? Same data (SplitMix64, docs/rng.md), same summation
 * order (docs/linalg.md section 2: blocks of 256, four lanes in a block, the tree over blocks, Cholesky in panels of 32; rewritten at v1-lanes),
 * so the result hash must equal linalg-check's; the time is the mean of REPS runs.
 *     cc -O3 -ffp-contract=off -o build/parity scripts/c_parity/parity.c
 *     cc -O3 -ffp-contract=fast -o build/parity_fma scripts/c_parity/parity.c     (ablation: the cost of the no-fma rule; bits differ)
 *     build/parity KERNEL N D SEED REPS        (KERNEL dot axpy gemv_n gemv_t syrk chol; ablations dot_lanes dot_chain)
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

static double lane_combine(double s0, double s1, double s2, double s3) { return (s0 + s1) + (s2 + s3); }
/* one block of m <= K terms, four lanes: term j to lane j mod 4 */
static double block_dot(const double *x, const double *y, long m) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0; long j = 0;
    for (; j + 4 <= m; j += 4) { s0 = s0 + x[j] * y[j]; s1 = s1 + x[j + 1] * y[j + 1]; s2 = s2 + x[j + 2] * y[j + 2]; s3 = s3 + x[j + 3] * y[j + 3]; }
    if (j < m) s0 = s0 + x[j] * y[j];
    if (j + 1 < m) s1 = s1 + x[j + 1] * y[j + 1];
    if (j + 2 < m) s2 = s2 + x[j + 2] * y[j + 2];
    return lane_combine(s0, s1, s2, s3);
}

static double dot(const double *x, const double *y, long n, double *st, int *lv) {
    if (n <= 0) return 0.0;
    long nb = (n - 1) / K + 1; int sp = 0; long b = 0;
    while (b < nb) {
        if (b + 2 <= nb && (b + 2) * K <= n) {  /* two full blocks together: eight chains */
            double a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
            const double *xa = x + b * K, *ya = y + b * K, *xb = xa + K, *yb = ya + K;
            for (long j = 0; j < K; j += 4) {
                a0 = a0 + xa[j] * ya[j]; a1 = a1 + xa[j + 1] * ya[j + 1]; a2 = a2 + xa[j + 2] * ya[j + 2]; a3 = a3 + xa[j + 3] * ya[j + 3];
                b0 = b0 + xb[j] * yb[j]; b1 = b1 + xb[j + 1] * yb[j + 1]; b2 = b2 + xb[j + 2] * yb[j + 2]; b3 = b3 + xb[j + 3] * yb[j + 3];
            }
            st[sp] = lane_combine(a0, a1, a2, a3); sp = settle(st, lv, sp, 1);
            st[sp] = lane_combine(b0, b1, b2, b3); sp = settle(st, lv, sp, 1);
            b += 2;
        } else {
            long m = n - b * K; if (m > K) m = K;
            st[sp] = block_dot(x + b * K, y + b * K, m); sp = settle(st, lv, sp, 1); b++;
        }
    }
    finish(st, sp, 1); return st[0];
}

/* ABLATIONS of the order rule (outputs differ from linalg-check; time only): the best scalar dot without blocks */
static double dot_lanes(const double *x, const double *y, long n) {  /* 8 accumulators over ADJACENT elements */
    double s[8] = {0, 0, 0, 0, 0, 0, 0, 0}; long i = 0;
    for (; i + 8 <= n; i += 8) for (int k = 0; k < 8; k++) s[k] = s[k] + x[i + k] * y[i + k];
    double t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (; i < n; i++) t = t + x[i] * y[i];
    return t;
}
#ifdef __ARM_NEON
#include <arm_neon.h>
static double dot_neon(const double *x, const double *y, long n) {  /* 4 vector accumulators (8 lanes), unfused; ablation of "no SIMD" */
    float64x2_t a0 = vdupq_n_f64(0), a1 = a0, a2 = a0, a3 = a0; long i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = vaddq_f64(a0, vmulq_f64(vld1q_f64(x + i), vld1q_f64(y + i)));         a1 = vaddq_f64(a1, vmulq_f64(vld1q_f64(x + i + 2), vld1q_f64(y + i + 2)));
        a2 = vaddq_f64(a2, vmulq_f64(vld1q_f64(x + i + 4), vld1q_f64(y + i + 4))); a3 = vaddq_f64(a3, vmulq_f64(vld1q_f64(x + i + 6), vld1q_f64(y + i + 6)));
    }
    double t = vaddvq_f64(vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3)));
    for (; i < n; i++) t = t + x[i] * y[i];
    return t;
}
#endif
static double dot_chain(const double *x, const double *y, long n) { double s = 0; for (long i = 0; i < n; i++) s = s + x[i] * y[i]; return s; }

static void axpy(double a, const double *x, double *y, long n) { for (long i = 0; i < n; i++) y[i] = y[i] + a * x[i]; }

static void gemv_n(const double *X, long n, long d, const double *w, double *y) {
    if (d > K) { for (long i = 0; i < n; i++) { /* block rule over j, then the tree */
        double st[64]; int lv[64]; int sp = 0; long nbk = (d - 1) / K + 1;
        for (long bk = 0; bk < nbk; bk++) { long m = d - bk * K; if (m > K) m = K; st[sp] = block_dot(X + i * d + bk * K, w + bk * K, m); sp = settle(st, lv, sp, 1); }
        finish(st, sp, 1); y[i] = st[0]; } return; }
    long i = 0;
    for (; i + 4 <= n; i += 4) {
        const double *r[4] = {X + i * d, X + (i + 1) * d, X + (i + 2) * d, X + (i + 3) * d};
        double z[4][4] = {{0}};
        long j = 0;
        for (; j + 4 <= d; j += 4) for (int q = 0; q < 4; q++) for (int l = 0; l < 4; l++) z[q][l] = z[q][l] + r[q][j + l] * w[j + l];
        for (int l = 0; l < 3; l++) if (j + l < d) for (int q = 0; q < 4; q++) z[q][l] = z[q][l] + r[q][j + l] * w[j + l];
        for (int q = 0; q < 4; q++) y[i + q] = lane_combine(z[q][0], z[q][1], z[q][2], z[q][3]);
    }
    for (; i < n; i++) y[i] = block_dot(X + i * d, w, d);
}

/* st holds 40 slots of width d and then four lane vectors of width d (44 * d doubles) */
static void gemv_t(const double *X, long n, long d, const double *r, double *g, double *st, int *lv) {
    long nb = (n - 1) / K + 1; int sp = 0; double *lane = st + 40 * d;
    for (long b = 0; b < nb; b++) {
        long lo = b * K, hi = lo + K; if (hi > n) hi = n;
        for (long j = 0; j < 4 * d; j++) lane[j] = 0.0;
        long i = lo;
        for (; i + 4 <= hi; i += 4) {
            const double *a0 = X + i * d, *a1 = a0 + d, *a2 = a1 + d, *a3 = a2 + d; double c0 = r[i], c1 = r[i + 1], c2 = r[i + 2], c3 = r[i + 3];
            for (long j = 0; j < d; j++) { lane[j] = lane[j] + a0[j] * c0; lane[d + j] = lane[d + j] + a1[j] * c1; lane[2 * d + j] = lane[2 * d + j] + a2[j] * c2; lane[3 * d + j] = lane[3 * d + j] + a3[j] * c3; }
        }
        for (int l = 0; i + l < hi; l++) { const double *a0 = X + (i + l) * d; double c0 = r[i + l]; for (long j = 0; j < d; j++) lane[l * d + j] = lane[l * d + j] + a0[j] * c0; }
        double *slot = st + (long)sp * d;
        for (long j = 0; j < d; j++) slot[j] = lane_combine(lane[j], lane[d + j], lane[2 * d + j], lane[3 * d + j]);
        sp = settle(st, lv, sp, d);
    }
    finish(st, sp, d); memcpy(g, st, d * 8);
}

static void syrk(const double *X, long n, long d, double *C, double *st, int *lv) {
    long nb = (n - 1) / K + 1; int sp = 0;
    for (long b = 0; b < nb; b++) {
        long lo = b * K, hi = lo + K; if (hi > n) hi = n;
        double *slot = st + (long)sp * d * d;
        for (long j0 = 0; j0 < d; j0 += 2) for (long k0 = j0; k0 < d; k0 += 2) {
            double c[2][2][4] = {{{0}}}; long jb = j0 + 2 <= d ? 2 : d - j0, kb = k0 + 2 <= d ? 2 : d - k0;
            for (long i = lo; i < hi; i++) {
                int l = (int)((i - lo) % 4);
                for (long u = 0; u < jb; u++) for (long v = 0; v < kb; v++) c[u][v][l] = c[u][v][l] + X[i * d + j0 + u] * X[i * d + k0 + v];
            }
            for (long u = 0; u < jb; u++) for (long v = 0; v < kb; v++) slot[(j0 + u) * d + k0 + v] = lane_combine(c[u][v][0], c[u][v][1], c[u][v][2], c[u][v][3]);
        }
        sp = settle(st, lv, sp, d * d);
    }
    finish(st, sp, d * d); memcpy(C, st, d * d * 8);
}

/* panels of 32, four lanes in a panel (docs/linalg.md 2.1); one entry at a time */
static double panel_sum(const double *L, long d, long i, long j, double aij) {
    double s = aij;
    for (long p0 = 0; p0 < j; p0 += 32) {
        long e = p0 + 32 < j ? p0 + 32 : j; double a[4] = {0, 0, 0, 0};
        for (long k = p0; k < e; k++) a[(k - p0) % 4] = a[(k - p0) % 4] + L[i * d + k] * L[j * d + k];
        s = s - lane_combine(a[0], a[1], a[2], a[3]);
    }
    return s;
}
static int chol(const double *A, double *L, long d) {
    for (long j = 0; j < d; j++) {
        double s = panel_sum(L, d, j, j, A[j * d + j]);
        if (!(s > 0.0)) return 1;
        double ljj = sqrt(s); L[j * d + j] = ljj;
        for (long i = j + 1; i < d; i++) L[i * d + j] = panel_sum(L, d, i, j, A[i * d + j]) / ljj;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 6) return 2;
    const char *kn = argv[1]; long n = atol(argv[2]), d = atol(argv[3]); uint64_t seed = atol(argv[4]); long reps = atol(argv[5]);
    uint64_t s1 = derive(seed, 1), s2 = derive(seed, 2), s3 = derive(seed, 3);
    double *st = aligned_alloc(64, (40 * (d > 0 ? d * d : 1) + 64 * d + 512) * 8 + 4096); int lv[64]; double t0, t1; uint64_t h = 0; long cnt = 0;
    if (!strcmp(kn, "dot")) {
        double *x = gen(n, s1), *y = gen(n, s2); double r = 0; t0 = now(); for (long q = 0; q < reps; q++) r = dot(x, y, n, st, lv); t1 = now(); h = hash(&r, 1); cnt = 1;
    } else if (!strcmp(kn, "dot_lanes") || !strcmp(kn, "dot_chain")) {
        double *x = gen(n, s1), *y = gen(n, s2); double r = 0; int lanes = !strcmp(kn, "dot_lanes");
        t0 = now(); for (long q = 0; q < reps; q++) r = lanes ? dot_lanes(x, y, n) : dot_chain(x, y, n); t1 = now(); h = hash(&r, 1); cnt = 1;
#ifdef __ARM_NEON
    } else if (!strcmp(kn, "dot_neon")) {
        double *x = gen(n, s1), *y = gen(n, s2); double r = 0; t0 = now(); for (long q = 0; q < reps; q++) r = dot_neon(x, y, n); t1 = now(); h = hash(&r, 1); cnt = 1;
#endif
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
    printf("hash %016llx\ncount %ld\nms %.9f\n", (unsigned long long)h, cnt, (t1 - t0) * 1000.0);
    return 0;
}
