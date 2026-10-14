/* SIMD ceiling experiment (docs/simd-ceiling.md): NEON kernels with a fixed lane order, the scalar reference of that order,
 * the bit-exactness check with mutants, and the timing driver. Not part of the library.
 *     cc -O3 -mcpu=native -fno-fast-math -ffp-contract=off -o build/simd scripts/simd/simd.c -lm
 *     build/simd check                                  bit equality NEON == scalar reference, all kernels, L, fma/unfused; mutants
 *     build/simd bench KERNEL N D SEED verify|noverify  one process times every variant of one cell (lines "res NAME SECONDS HASH VER")
 * KERNEL: dot axpy gemv_n gemv_t syrk chol */
#include <arm_neon.h>
#define main parity_main
#include "../c_parity/parity.c"
#undef main
#pragma STDC FP_CONTRACT OFF

typedef struct {
    int L, fm;
    double (*dot)(const double *, const double *, long, double *, int *);
    void (*gemv_n)(const double *, long, long, const double *, double *, double *, int *);
    void (*gemv_t)(const double *, long, long, const double *, double *, double *, int *);
    void (*syrk)(const double *, long, long, double *, double *, int *);
    int (*chol)(const double *, long, double *, double *);
} Impl;

#define LN 1
#define FM 0
#include "kern.h"
#undef FM
#define FM 1
#include "kern.h"
#undef FM
#undef LN
#define LN 2
#define FM 0
#include "kern.h"
#undef FM
#define FM 1
#include "kern.h"
#undef FM
#undef LN
#define LN 4
#define FM 0
#include "kern.h"
#undef FM
#define FM 1
#include "kern.h"
#undef FM
#undef LN
#define LN 8
#define FM 0
#include "kern.h"
#undef FM
#define FM 1
#include "kern.h"
#undef FM
#undef LN
#define LN 16
#define FM 0
#include "kern.h"
#undef FM
#define FM 1
#include "kern.h"
#undef FM
#undef LN

#define I1(F) {1, F, NULL, NULL, gemv_t_L1_f##F, syrk_L1_f##F, chol_L1_f##F}
#define IN(N, F) {N, F, dot_L##N##_f##F, gemv_n_L##N##_f##F, gemv_t_L##N##_f##F, syrk_L##N##_f##F, chol_L##N##_f##F}
static const Impl TAB[] = { I1(0), I1(1), IN(2, 0), IN(2, 1), IN(4, 0), IN(4, 1), IN(8, 0), IN(8, 1), IN(16, 0), IN(16, 1) };
#define NTAB ((int)(sizeof TAB / sizeof TAB[0]))

/* ------------------------------------------------------------------------------------------------ axpy (no order) */
static void axpy_neon(double a, const double *x, double *y, long n, int fm) {
    float64x2_t av = vdupq_n_f64(a); long i = 0;
    if (fm) {
        for (; i + 8 <= n; i += 8) for (int r = 0; r < 4; r++) vst1q_f64(y + i + 2 * r, vfmaq_f64(vld1q_f64(y + i + 2 * r), av, vld1q_f64(x + i + 2 * r)));
        for (; i < n; i++) y[i] = fma(a, x[i], y[i]);
    } else {
        for (; i + 8 <= n; i += 8) for (int r = 0; r < 4; r++) vst1q_f64(y + i + 2 * r, vaddq_f64(vld1q_f64(y + i + 2 * r), vmulq_f64(av, vld1q_f64(x + i + 2 * r))));
        for (; i < n; i++) y[i] = y[i] + a * x[i];
    }
}
static void axpy_ref(double a, const double *x, double *y, long n, int fm) { for (long i = 0; i < n; i++) y[i] = fm ? fma(a, x[i], y[i]) : y[i] + a * x[i]; }

/* "free" dot: 16 lanes, no blocks, one tree at the end (bits differ from every fixed order) */
static double dot_free16(const double *x, const double *y, long n) {
    float64x2_t a[8]; for (int r = 0; r < 8; r++) a[r] = vdupq_n_f64(0.0);
    long j = 0;
    for (; j + 16 <= n; j += 16) for (int r = 0; r < 8; r++) a[r] = vaddq_f64(a[r], vmulq_f64(vld1q_f64(x + j + 2 * r), vld1q_f64(y + j + 2 * r)));
    for (int c = 8; c > 1; c /= 2) for (int i = 0; i < c / 2; i++) a[i] = vpaddq_f64(a[2 * i], a[2 * i + 1]);
    double t = vaddvq_f64(a[0]);
    for (; j < n; j++) t = t + x[j] * y[j];
    return t;
}

/* ------------------------------------------------------------------------------------------------ reference (the definition) */
/* g_mut: 0 none; 1 lane count doubled; 2 lanes combined by a left fold instead of the tree; 3 element -> lane map (i/2) mod L;
 *        4 block size 128 instead of 256 (reductions over rows / j); 5 chol panel width 16 instead of 32 */
static int g_mut = 0;
static double combine(double *s, int c) {
    if (g_mut == 2) { double t = s[0]; for (int l = 1; l < c; l++) t = t + s[l]; return t; }
    while (c > 1) { for (int i = 0; i < c / 2; i++) s[i] = s[2 * i] + s[2 * i + 1]; c /= 2; }
    return s[0];
}
static double lane_sum(const double *x, long sx, const double *y, long sy, long m, int L, int fm) {
    int LL = (g_mut == 1) ? 2 * L : L; double s[64]; for (int l = 0; l < LL; l++) s[l] = 0.0;
    for (long i = 0; i < m; i++) {
        int l = (g_mut == 3) ? (int)((i / 2) % LL) : (int)(i % LL);
        double a = x[i * sx], b = y[i * sy]; s[l] = fm ? fma(a, b, s[l]) : s[l] + a * b;
    }
    return combine(s, LL);
}
static double ref_dot(const double *x, long sx, const double *y, long sy, long n, int L, int fm) {
    if (n <= 0) return 0.0;
    long Kk = (g_mut == 4) ? 128 : 256, nb = (n - 1) / Kk + 1; double *p = malloc(nb * 8);
    for (long b = 0; b < nb; b++) { long m = n - b * Kk; if (m > Kk) m = Kk; p[b] = lane_sum(x + b * Kk * sx, sx, y + b * Kk * sy, sy, m, L, fm); }
    long cnt = nb;
    while (cnt > 1) { for (long i = 0; i < cnt / 2; i++) p[i] = p[2 * i] + p[2 * i + 1]; if (cnt & 1) p[cnt / 2] = p[cnt - 1]; cnt = (cnt + 1) / 2; }
    double r = p[0]; free(p); return r;
}
static void ref_gemv_n(const double *X, long n, long d, const double *w, double *y, int L, int fm) { for (long i = 0; i < n; i++) y[i] = ref_dot(X + i * d, 1, w, 1, d, L, fm); }
static void ref_gemv_t(const double *X, long n, long d, const double *r, double *g, int L, int fm) { for (long j = 0; j < d; j++) g[j] = ref_dot(X + j, d, r, 1, n, L, fm); }
static void ref_syrk(const double *X, long n, long d, double *C, int L, int fm) {
    memset(C, 0, d * d * 8);
    for (long j = 0; j < d; j++) for (long k = j; k < d; k++) C[j * d + k] = ref_dot(X + j, d, X + k, d, n, L, fm);
}
static int ref_chol(const double *A, long d, double *Lm, int L, int fm) {
    long NBm = (g_mut == 5) ? 16 : 32; memset(Lm, 0, d * d * 8);
    for (long j = 0; j < d; j++) for (long i = j; i < d; i++) {
        double s = A[i * d + j]; long c0 = j / NBm * NBm;
        for (long p0 = 0; p0 < c0; p0 += NBm) s = s - lane_sum(Lm + i * d + p0, 1, Lm + j * d + p0, 1, NBm, L, fm);
        s = s - lane_sum(Lm + i * d + c0, 1, Lm + j * d + c0, 1, j - c0, L, fm);
        if (i == j) { if (!(s > 0.0)) return 1; Lm[i * d + j] = sqrt(s); } else Lm[i * d + j] = s / Lm[j * d + j];
    }
    return 0;
}

/* ------------------------------------------------------------------------------------------------ the check */
static uint64_t rs;
static uint64_t rnext(void) { rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return rs * 0x2545F4914F6CDD1DULL; }
static void fill(double *a, long n, int wide) {  /* wide magnitudes (2^-20 .. 2^20) so that the order of the additions shows; some exact zeros */
    for (long i = 0; i < n; i++) {
        uint64_t r = rnext(); double u = (double)(r >> 11) / 4503599627370496.0 - 1.0;
        if (wide) { u = ldexp(u, (int)((r >> 3) % 41) - 20); if ((r & 0x3f) == 0) u = 0.0; }
        a[i] = u;
    }
}
static int same(const double *a, const double *b, long n) { return memcmp(a, b, n * 8) == 0; }
static void make_spd(double *A, long d) {
    double *E = malloc(d * d * 8); fill(E, d * d, 0);
    for (long i = 0; i < d; i++) for (long j = 0; j < d; j++) A[i * d + j] = (i == j) ? ((double)d + 1.0) + E[i * d + j] : 0.5 * (E[i * d + j] + E[j * d + i]);
    free(E);
}
enum { KD, KGN, KGT, KSY, KCH, NK };
static const char *KNAME[NK] = {"dot", "gemv_n", "gemv_t", "syrk", "chol"};
typedef struct { long a, b; } Shape;
static const long DOT_N[] = {0, 1, 2, 3, 15, 16, 17, 31, 255, 256, 257, 511, 512, 700, 1000, 4099, 70001};
static const Shape GN_S[] = {{1, 1}, {5, 3}, {7, 20}, {9, 21}, {9, 64}, {13, 128}, {13, 129}, {13, 256}, {6, 257}, {4, 600}, {10, 17}, {3, 1000}};
static const Shape GT_S[] = {{0, 5}, {1, 1}, {3, 7}, {16, 20}, {255, 21}, {256, 64}, {257, 3}, {700, 20}, {700, 64}, {1000, 21}, {1000, 33}, {513, 256}, {300, 300}, {1000, 1}};
static const long CH_D[] = {1, 2, 5, 16, 31, 32, 33, 64, 100, 255, 256};

/* one case: returns 1 when the NEON result differs from the reference under mutant `mut` */
static int run_case(int kernel, const Impl *I, long a, long b, int mut) {
    int differ = 0; g_mut = mut; int L = I->L, fm = I->fm;
    rs = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(kernel * 1000003 + a * 7919 + b * 104729 + 1);
    int lv[64];
    if (kernel == KD) {
        long n = a; double *x = malloc(n * 8 + 64), *y = malloc(n * 8 + 64), *st = calloc(64, 8); fill(x, n, 1); fill(y, n, 1);
        double r = I->dot(x, y, n, st, lv), q = ref_dot(x, 1, y, 1, n, L, fm); differ = !same(&r, &q, 1); free(x); free(y); free(st);
    } else if (kernel == KGN) {
        long n = a, d = b; double *X = malloc(n * d * 8 + 64), *w = malloc(d * 8 + 64), *y = calloc(n + 1, 8), *q = calloc(n + 1, 8), *st = calloc(64 * 2, 8);
        fill(X, n * d, 1); fill(w, d, 1); I->gemv_n(X, n, d, w, y, st, lv); ref_gemv_n(X, n, d, w, q, L, fm); differ = !same(y, q, n);
        free(X); free(w); free(y); free(q); free(st);
    } else if (kernel == KGT) {
        long n = a, d = b; double *X = malloc(n * d * 8 + 64), *r = malloc(n * 8 + 64), *y = calloc(d + 1, 8), *q = calloc(d + 1, 8), *st = calloc(64 * d + 64, 8);
        fill(X, n * d, 1); fill(r, n, 1); I->gemv_t(X, n, d, r, y, st, lv); ref_gemv_t(X, n, d, r, q, L, fm); differ = !same(y, q, d);
        free(X); free(r); free(y); free(q); free(st);
    } else if (kernel == KSY) {
        long n = a, d = b; double *X = malloc(n * d * 8 + 64), *C = calloc(d * d + 1, 8), *q = calloc(d * d + 1, 8), *st = calloc(40 * d * d + 64, 8);
        fill(X, n * d, 1); I->syrk(X, n, d, C, st, lv); ref_syrk(X, n, d, q, L, fm);
        for (long j = 0; j < d; j++) for (long k = j; k < d; k++) if (!same(C + j * d + k, q + j * d + k, 1)) differ = 1;
        free(X); free(C); free(q); free(st);
    } else {
        long d = a, dp = (d + 15) / 16 * 16; double *A = malloc(d * d * 8 + 64), *Lr = calloc(d * d + 1, 8);
        double *W = calloc((dp + 16) * dp + 64, 8), *Pt = calloc(32 * (dp + 16) + 64, 8); make_spd(A, d);
        int e1 = I->chol(A, d, W, Pt), e2 = ref_chol(A, d, Lr, L, fm);
        if (e1 != e2) differ = 1;
        for (long i = 0; i < d; i++) for (long j = 0; j <= i; j++) if (!same(W + i * dp + j, Lr + i * d + j, 1)) differ = 1;
        free(A); free(Lr); free(W); free(Pt);
    }
    g_mut = 0; return differ;
}
static int run_kernel(int kernel, const Impl *I, int mut) {  /* 1 if any shape differs */
    int dif = 0;
    if (kernel == KD) for (size_t i = 0; i < sizeof DOT_N / 8; i++) dif |= run_case(kernel, I, DOT_N[i], 0, mut);
    if (kernel == KGN) for (size_t i = 0; i < sizeof GN_S / sizeof(Shape); i++) dif |= run_case(kernel, I, GN_S[i].a, GN_S[i].b, mut);
    if (kernel == KGT || kernel == KSY) for (size_t i = 0; i < sizeof GT_S / sizeof(Shape); i++) {
        if (kernel == KSY && GT_S[i].a * GT_S[i].b * GT_S[i].b > 4000000) continue;
        dif |= run_case(kernel, I, GT_S[i].a, GT_S[i].b, mut);
    }
    if (kernel == KSY) { static const Shape ex[] = {{255, 256}, {700, 256}, {257, 64}, {700, 20}}; for (int i = 0; i < 4; i++) dif |= run_case(kernel, I, ex[i].a, ex[i].b, mut); }
    if (kernel == KCH) for (size_t i = 0; i < sizeof CH_D / 8; i++) dif |= run_case(kernel, I, CH_D[i], 0, mut);
    return dif;
}
static int check_all(void) {
    int bad = 0;
    printf("== bit equality NEON vs scalar reference of the same order (0 = identical on every shape)\n");
    for (int t = 0; t < NTAB; t++) {
        const Impl *I = &TAB[t]; printf("L=%-2d %-6s:", I->L, I->fm ? "fma" : "unfused");
        for (int k = 0; k < NK; k++) {
            if (I->L == 1 && (k == KD || k == KGN)) { printf("  %s=n/a", KNAME[k]); continue; }
            int d = run_kernel(k, I, 0); bad += d; printf("  %s=%s", KNAME[k], d ? "DIFFER" : "identical");
        }
        printf("\n");
    }
    { double x[1003], y[1003], y2[1003]; rs = 77; fill(x, 1003, 1); fill(y, 1003, 1); int ax = 0;
      for (int fm = 0; fm < 2; fm++) { memcpy(y2, y, sizeof y); double a = 0.3712; memcpy(y2, y, sizeof y); axpy_neon(a, x, y2, 1003, fm); double q[1003]; memcpy(q, y, sizeof y); axpy_ref(a, x, q, 1003, fm); ax |= !same(y2, q, 1003); }
      bad += ax; printf("axpy (no order): unfused+fma %s\n", ax ? "DIFFER" : "identical"); }
    printf("== mutants of the reference (must differ from the NEON result on at least one shape; EQ = provably equivalent)\n");
    static const char *MN[] = {"", "lanes x2", "left-fold combine", "lane map (i/2)%L", "block 128", "panel 16 (chol)"};
    for (int t = 0; t < NTAB; t++) {
        const Impl *I = &TAB[t]; printf("L=%-2d %-6s:", I->L, I->fm ? "fma" : "unfused");
        for (int k = 0; k < NK; k++) {
            if (I->L == 1 && (k == KD || k == KGN)) continue;
            for (int m = 1; m <= 5; m++) {
                if (m == 5 && k != KCH) continue;
                if (m == 4 && k == KCH) continue;
                int det = run_kernel(k, I, m);
                int equiv = (m == 2 && I->L <= 2) || (m == 3 && I->L == 1);  /* left fold of <=2 lanes = tree; (i/2)%1 = i%1 */
                const char *res = det ? "killed" : (equiv ? "EQ" : "SURVIVES");
                if (!det && !equiv) bad++;
                printf(" %s/%s=%s", KNAME[k], MN[m], res);
            }
        }
        printf("\n");
    }
    printf(bad ? "CHECK FAILED (%d)\n" : "CHECK PASSED (%d bad)\n", bad);
    return bad != 0;
}

/* ------------------------------------------------------------------------------------------------ bench */
#define TIMEIT(secs, stmt) do { double t0_ = now(); stmt; double t1_ = now() - t0_; long R_ = 1; double tot_; \
    if (t1_ >= 0.04) { t0_ = now(); stmt; tot_ = now() - t0_; } \
    else { R_ = (long)(0.04 / (t1_ > 1e-9 ? t1_ : 1e-9)) + 1; t0_ = now(); for (long q_ = 0; q_ < R_; q_++) { stmt; } tot_ = now() - t0_; } \
    secs = tot_ / (double)R_; } while (0)

static uint64_t hash_tri(const double *M, long ld, long d, int lower) {
    double *c = calloc(d * d + 1, 8);
    for (long i = 0; i < d; i++) for (long j = 0; j < d; j++) if (lower ? (j <= i) : (j >= i)) c[i * d + j] = M[i * ld + j];
    uint64_t h = hash(c, d * d); free(c); return h;
}
static void out(const char *name, double sec, uint64_t h, const char *ver) { printf("res %s %.9e %016llx %s\n", name, sec, (unsigned long long)h, ver); fflush(stdout); }

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "check")) return check_all();
    if (argc < 7 || strcmp(argv[1], "bench")) return 2;
    const char *kn = argv[2]; long n = atol(argv[3]), d = atol(argv[4]); uint64_t seed = atol(argv[5]); int verify = !strcmp(argv[6], "verify");
    uint64_t s1 = derive(seed, 1), s2 = derive(seed, 2), s3 = derive(seed, 3);
    double stk = (d > 0 ? (double)d * d : 1.0) * 40 * 8 + 4096; double *st = calloc(1, (size_t)stk), *st0 = aligned_alloc(64, ((size_t)stk + 63) / 64 * 64); int lv[64]; double sec;
    char vb[16];
    if (!strcmp(kn, "dot")) {
        double *x = gen(n, s1), *y = gen(n, s2), r = 0;
        TIMEIT(sec, r = dot(x, y, n, st0, lv)); out("CS", sec, hash(&r, 1), "-");
        for (int t = 0; t < NTAB; t++) { const Impl *I = &TAB[t]; if (!I->dot) continue;
            TIMEIT(sec, r = I->dot(x, y, n, st, lv));
            const char *v = "noverify"; if (verify) { double q = ref_dot(x, 1, y, 1, n, I->L, I->fm); v = (q == r && !memcmp(&q, &r, 8)) ? "ok" : "FAIL"; }
            snprintf(vb, sizeof vb, "%c%d", I->fm ? 'F' : 'N', I->L); out(vb, sec, hash(&r, 1), v); }
        TIMEIT(sec, r = dot_free16(x, y, n)); out("free16", sec, hash(&r, 1), "-");
    } else if (!strcmp(kn, "axpy")) {
        double *x = gen(n, s1), *y = gen(n, s2), a = element(s3, 0), *y0 = gen(n, s2);
        TIMEIT(sec, axpy(a, x, y, n)); out("CS", sec, 0, "-");
        memcpy(y, y0, n * 8); TIMEIT(sec, axpy_neon(a, x, y, n, 0)); out("N", sec, 0, "-");
        memcpy(y, y0, n * 8); TIMEIT(sec, axpy_neon(a, x, y, n, 1)); out("F", sec, 0, "-");
    } else if (!strcmp(kn, "gemv_n")) {
        double *X = gen(n * d, s1), *w = gen(d, s2), *y = calloc(n + 8, 8), *q = calloc(n + 8, 8);
        TIMEIT(sec, gemv_n(X, n, d, w, y)); out("CS", sec, hash(y, n), "-");
        for (int t = 0; t < NTAB; t++) { const Impl *I = &TAB[t]; if (!I->gemv_n) continue;
            TIMEIT(sec, I->gemv_n(X, n, d, w, y, st, lv));
            const char *v = "noverify"; if (verify) { ref_gemv_n(X, n, d, w, q, I->L, I->fm); v = same(y, q, n) ? "ok" : "FAIL"; }
            snprintf(vb, sizeof vb, "%c%d", I->fm ? 'F' : 'N', I->L); out(vb, sec, hash(y, n), v); }
    } else if (!strcmp(kn, "gemv_t")) {
        double *X = gen(n * d, s1), *r = gen(n, s2), *g = calloc(d + 8, 8), *q = calloc(d + 8, 8);
        TIMEIT(sec, gemv_t(X, n, d, r, g, st0, lv)); out("CS", sec, hash(g, d), "-");
        for (int t = 0; t < NTAB; t++) { const Impl *I = &TAB[t];
            TIMEIT(sec, I->gemv_t(X, n, d, r, g, st, lv));
            const char *v = "noverify"; if (verify && (double)n * d < 3e8) { ref_gemv_t(X, n, d, r, q, I->L, I->fm); v = same(g, q, d) ? "ok" : "FAIL"; }
            snprintf(vb, sizeof vb, "%c%d", I->fm ? 'F' : 'N', I->L); out(vb, sec, hash(g, d), v); }
    } else if (!strcmp(kn, "syrk")) {
        double *X = gen(n * d, s1), *C = calloc(d * d + 8, 8), *full = calloc(d * d + 8, 8), *q = calloc(d * d + 8, 8);
        TIMEIT(sec, syrk(X, n, d, full, st0, lv)); out("CS", sec, hash_tri(full, d, d, 0), "-");
        for (int t = 0; t < NTAB; t++) { const Impl *I = &TAB[t];
            TIMEIT(sec, I->syrk(X, n, d, C, st, lv));
            const char *v = "noverify"; if (verify && (double)n * d * d / 2 < 1.5e9) { ref_syrk(X, n, d, q, I->L, I->fm); v = (hash_tri(C, d, d, 0) == hash_tri(q, d, d, 0)) ? "ok" : "FAIL"; }
            snprintf(vb, sizeof vb, "%c%d", I->fm ? 'F' : 'N', I->L); out(vb, sec, hash_tri(C, d, d, 0), v); }
    } else if (!strcmp(kn, "chol")) {
        double *A = calloc(n * n + 8, 8), *Lc = calloc(n * n + 8, 8), *q = calloc(n * n + 8, 8);
        double *E = gen(n * n, s1);
        for (long i = 0; i < n; i++) for (long j = 0; j < n; j++) A[i * n + j] = (i == j) ? ((double)n + 1.0) + E[i * n + j] : 0.5 * (E[i * n + j] + E[j * n + i]);
        long dp = (n + 15) / 16 * 16; double *W = calloc((dp + 16) * dp + 64, 8), *Pt = calloc(32 * (dp + 16) + 64, 8);
        TIMEIT(sec, chol(A, Lc, n)); out("CS", sec, hash(Lc, n * n), "-");
        for (int t = 0; t < NTAB; t++) { const Impl *I = &TAB[t];
            TIMEIT(sec, I->chol(A, n, W, Pt));
            const char *v = "noverify"; if (verify) { ref_chol(A, n, q, I->L, I->fm); v = (hash_tri(W, dp, n, 1) == hash(q, n * n)) ? "ok" : "FAIL"; }
            snprintf(vb, sizeof vb, "%c%d", I->fm ? 'F' : 'N', I->L); out(vb, sec, hash_tri(W, dp, n, 1), v); }
    } else return 2;
    return 0;
}
