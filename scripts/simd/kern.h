/* NEON kernels with the fixed lane order of docs/simd-ceiling.md section 2. Included once per (LN, FM) by simd.c:
 *   LN = number of lane accumulators per reduction (1, 2, 4, 8, 16); FM = 0 unfused (vmulq + vaddq), 1 = vfmaq_f64 in the lane
 *   accumulation only (changes bits; the reference uses fma() the same way). Names get the suffix _L<LN>_f<FM>.
 * Uses K, settle(), finish() from scripts/c_parity/parity.c (included by simd.c). Intrinsics experiment: not shipped. */
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define NM(x) CAT(x, CAT(_L, CAT(LN, CAT(_f, FM))))
#define UNR _Pragma("clang loop unroll(full)")
#define AI static inline __attribute__((always_inline))
#if FM
#define VMAC(acc, a, b) vfmaq_f64((acc), (a), (b))
#define SMAC(acc, a, b) fma((a), (b), (acc))
#else
#define VMAC(acc, a, b) vaddq_f64((acc), vmulq_f64((a), (b)))
#define SMAC(acc, a, b) ((acc) + (a) * (b))
#endif
#define NR (LN / 2)
#define QL (4 * LN)

/* per-L register shapes (tuned by hand; accumulators = about 16 registers) */
#if LN == 1
#define MU 4   /* syrk / chol trailing tile: MU rows x NV vectors */
#define NV 4
#define RV 8   /* chol panel: row vectors in flight */
#define RW 4   /* gemv_n rows in flight */
#elif LN == 2
#define MU 4
#define NV 2
#define RV 8
#define RW 4
#elif LN == 4
#define MU 2
#define NV 2
#define RV 4
#define RW 4
#elif LN == 8
#define MU 2
#define NV 1
#define RV 2
#define RW 4
#else
#define MU 1
#define NV 1
#define RV 1
#define RW 2
#endif
#define CW (2 * NV)    /* tile columns */

#if LN >= 2
/* lane tree of NR registers: pairwise vpaddq keeps q_i = s_2i + s_2i+1 level by level, the last vaddv is the final level */
AI double NM(vsum)(float64x2_t *a) {
    UNR for (int c = NR; c > 1; c /= 2) { UNR for (int i = 0; i < c / 2; i++) a[i] = vpaddq_f64(a[2 * i], a[2 * i + 1]); }
    return vaddvq_f64(a[0]);
}

static double NM(dot)(const double *x, const double *y, long n, double *st, int *lv) {
    if (n <= 0) return 0.0;
    long nb = (n - 1) / K + 1; int sp = 0;
    for (long b = 0; b < nb; b++) {
        long lo = b * K, m = n - lo; if (m > K) m = K;
        const double *xb = x + lo, *yb = y + lo;
        float64x2_t a[NR];
        UNR for (int r = 0; r < NR; r++) a[r] = vdupq_n_f64(0.0);
        long j = 0;
        for (; j + LN <= m; j += LN) { UNR for (int r = 0; r < NR; r++) a[r] = VMAC(a[r], vld1q_f64(xb + j + 2 * r), vld1q_f64(yb + j + 2 * r)); }
        if (j < m) {  /* zero-padded last vector step: adding +0.0 (or fma(0,0,acc)) leaves every lane unchanged, so it equals "lane stays as is" */
            double tx[LN], ty[LN]; for (int t = 0; t < LN; t++) { tx[t] = 0.0; ty[t] = 0.0; }
            for (long t = 0; j + t < m; t++) { tx[t] = xb[j + t]; ty[t] = yb[j + t]; }
            UNR for (int r = 0; r < NR; r++) a[r] = VMAC(a[r], vld1q_f64(tx + 2 * r), vld1q_f64(ty + 2 * r));
        }
        st[sp] = NM(vsum)(a); sp = settle(st, lv, sp, 1);
    }
    finish(st, sp, 1); return st[0];
}

AI void NM(gv_rows)(const double *X, long d, const double *w, double *y, const int RWc) {
    float64x2_t a[RW][NR];
    UNR for (int q = 0; q < RW; q++) { UNR for (int r = 0; r < NR; r++) a[q][r] = vdupq_n_f64(0.0); }
    long j = 0;
    for (; j + LN <= d; j += LN) {
        float64x2_t wv[NR];
        UNR for (int r = 0; r < NR; r++) wv[r] = vld1q_f64(w + j + 2 * r);
        UNR for (int q = 0; q < RWc; q++) { UNR for (int r = 0; r < NR; r++) a[q][r] = VMAC(a[q][r], vld1q_f64(X + q * d + j + 2 * r), wv[r]); }
    }
    if (j < d) {
        double tw[LN]; for (int t = 0; t < LN; t++) tw[t] = 0.0;
        for (long t = 0; j + t < d; t++) tw[t] = w[j + t];
        float64x2_t wv[NR];
        UNR for (int r = 0; r < NR; r++) wv[r] = vld1q_f64(tw + 2 * r);
        UNR for (int q = 0; q < RWc; q++) {
            double tx[LN]; for (int t = 0; t < LN; t++) tx[t] = 0.0;
            for (long t = 0; j + t < d; t++) tx[t] = X[q * d + j + t];
            UNR for (int r = 0; r < NR; r++) a[q][r] = VMAC(a[q][r], vld1q_f64(tx + 2 * r), wv[r]);
        }
    }
    UNR for (int q = 0; q < RWc; q++) y[q] = NM(vsum)(a[q]);
}

static void NM(gemv_n)(const double *X, long n, long d, const double *w, double *y, double *st, int *lv) {
    if (d > K) { for (long i = 0; i < n; i++) y[i] = NM(dot)(X + i * d, w, d, st, lv); return; }
    long i = 0;
    for (; i + RW <= n; i += RW) NM(gv_rows)(X + i * d, d, w, y + i, RW);
    for (; i < n; i++) NM(gv_rows)(X + i * d, d, w, y + i, 1);
}
#endif /* LN >= 2 */

/* gemv_t, row streaming: the LN lane accumulator rows ws[LN][d] stay in L1; each row of X (sequential memory) is added into the lane
 * row i mod LN. Same additions in the same order as the lane rule. */
static void NM(gemv_t)(const double *X, long n, long d, const double *r, double *g, double *st, int *lv) {
    if (n <= 0) { memset(g, 0, d * 8); return; }
    long nb = (n - 1) / K + 1; int sp = 0; const long dv = d / 2 * 2;
    double *ws = malloc((size_t)LN * (d + 8) * 8 + 64);
    for (long b = 0; b < nb; b++) {
        long lo = b * K, m = n - lo; if (m > K) m = K;
        double *slot = st + (long)sp * d;
        memset(ws, 0, (size_t)LN * d * 8);
        long i = 0;
        const long qmax = (((long)LN * d * 8) % 4096 != 0) ? m : 0;  /* rows LN apart at a multiple of 4 KB alias in L1 and in the store forwarding: row streaming then */
        for (; i + QL <= qmax; i += QL) {  /* Q rows of the same lane (i+l, i+l+LN, ...) are added in turn: one load and one store of the accumulator row per Q rows */
            for (int l = 0; l < LN; l++) {
                double *acc = ws + (long)l * d; const double *r0 = X + (lo + i + l) * d, *r1 = r0 + (long)LN * d, *r2 = r1 + (long)LN * d, *r3 = r2 + (long)LN * d;
                const double *rr = r + lo + i + l; float64x2_t c0 = vdupq_n_f64(rr[0]), c1 = vdupq_n_f64(rr[LN]), c2 = vdupq_n_f64(rr[2 * LN]), c3 = vdupq_n_f64(rr[3 * LN]);
                for (long j = 0; j < dv; j += 2) {
                    float64x2_t v = vld1q_f64(acc + j);
                    v = VMAC(v, vld1q_f64(r0 + j), c0); v = VMAC(v, vld1q_f64(r1 + j), c1); v = VMAC(v, vld1q_f64(r2 + j), c2); v = VMAC(v, vld1q_f64(r3 + j), c3);
                    vst1q_f64(acc + j, v);
                }
                if (dv < d) { double v = acc[dv]; v = SMAC(v, r0[dv], rr[0]); v = SMAC(v, r1[dv], rr[LN]); v = SMAC(v, r2[dv], rr[2 * LN]); v = SMAC(v, r3[dv], rr[3 * LN]); acc[dv] = v; }
            }
        }
        for (; i < m; i++) {
            double *acc = ws + (long)(i % LN) * d; const double *row = X + (lo + i) * d; double c = r[lo + i]; float64x2_t cc = vdupq_n_f64(c);
            long j = 0;
            for (; j + 8 <= dv; j += 8) { UNR for (int u = 0; u < 4; u++) vst1q_f64(acc + j + 2 * u, VMAC(vld1q_f64(acc + j + 2 * u), vld1q_f64(row + j + 2 * u), cc)); }
            for (; j < dv; j += 2) vst1q_f64(acc + j, VMAC(vld1q_f64(acc + j), vld1q_f64(row + j), cc));
            if (dv < d) acc[dv] = SMAC(acc[dv], row[dv], c);
        }
        for (long j = 0; j + 2 <= d; j += 2) {
            float64x2_t a[LN];
            UNR for (int l = 0; l < LN; l++) a[l] = vld1q_f64(ws + (long)l * d + j);
            UNR for (int c = LN; c > 1; c /= 2) { UNR for (int q = 0; q < c / 2; q++) a[q] = vaddq_f64(a[2 * q], a[2 * q + 1]); }
            vst1q_f64(slot + j, a[0]);
        }
        if (dv < d) { double s[LN]; for (int l = 0; l < LN; l++) s[l] = ws[(long)l * d + dv];
            for (int c = LN; c > 1; c /= 2) for (int q = 0; q < c / 2; q++) s[q] = s[2 * q] + s[2 * q + 1];
            slot[dv] = s[0]; }
        sp = settle(st, lv, sp, d);
    }
    free(ws);
    finish(st, sp, d); memcpy(g, st, d * 8);
}

AI int NM(covered)(long j, long k, long d) { return (j / MU * MU + MU <= d) && (k / CW * CW + CW <= d); }

/* syrk: for d > 32, per block of K rows the block is packed into 8-column panels P[panel][row][8] (contiguous, no power-of-two row stride, so no
 * L1 conflicts at d = 256; the packing is a copy and changes no value); tiles read the panels. */
static void NM(syrk)(const double *X, long n, long d, double *C, double *st, int *lv) {
    if (n <= 0) { memset(C, 0, d * d * 8); return; }
    long nb = (n - 1) / K + 1; int sp = 0; const long np = (d + 7) / 8, packed = d > 32, rs = packed ? 8 : d;
    double *P = malloc((size_t)np * K * 8 * 8 + 64);
    for (long b = 0; b < nb; b++) {
        long lo = b * K, m = n - lo; if (m > K) m = K;
        double *slot = st + (long)sp * d * d; const double *xb = X + lo * d;
        if (packed) for (long pj = 0; pj < np; pj++) { long w = d - pj * 8; if (w > 8) w = 8; double *dst = P + pj * K * 8; for (long i = 0; i < m; i++) memcpy(dst + i * 8, xb + i * d + pj * 8, w * 8); }
        for (long j0 = 0; j0 + MU <= d; j0 += MU) for (long k0 = j0 / CW * CW; k0 + CW <= d; k0 += CW) {
            float64x2_t acc[LN][MU][NV];
            UNR for (int l = 0; l < LN; l++) { UNR for (int u = 0; u < MU; u++) { UNR for (int v = 0; v < NV; v++) acc[l][u][v] = vdupq_n_f64(0.0); } }
            const double *pj_ = packed ? P + (j0 / 8) * K * 8 + (j0 % 8) : xb + j0, *pk_ = packed ? P + (k0 / 8) * K * 8 + (k0 % 8) : xb + k0;
            long i = 0;
            for (; i + LN <= m; i += LN) {
                UNR for (int l = 0; l < LN; l++) {
                    const double *rj = pj_ + (i + l) * rs, *rk = pk_ + (i + l) * rs; float64x2_t q[NV];
                    UNR for (int v = 0; v < NV; v++) q[v] = vld1q_f64(rk + 2 * v);
                    UNR for (int u = 0; u < MU; u++) { float64x2_t p = vdupq_n_f64(rj[u]); UNR for (int v = 0; v < NV; v++) acc[l][u][v] = VMAC(acc[l][u][v], p, q[v]); }
                }
            }
            UNR for (int l = 0; l < LN; l++) if (i + l < m) {
                const double *rj = pj_ + (i + l) * rs, *rk = pk_ + (i + l) * rs; float64x2_t q[NV];
                UNR for (int v = 0; v < NV; v++) q[v] = vld1q_f64(rk + 2 * v);
                UNR for (int u = 0; u < MU; u++) { float64x2_t p = vdupq_n_f64(rj[u]); UNR for (int v = 0; v < NV; v++) acc[l][u][v] = VMAC(acc[l][u][v], p, q[v]); }
            }
            UNR for (int c = LN; c > 1; c /= 2) { UNR for (int q = 0; q < c / 2; q++) { UNR for (int u = 0; u < MU; u++) { UNR for (int v = 0; v < NV; v++) acc[q][u][v] = vaddq_f64(acc[2 * q][u][v], acc[2 * q + 1][u][v]); } } }
            UNR for (int u = 0; u < MU; u++) { UNR for (int v = 0; v < NV; v++) vst1q_f64(slot + (j0 + u) * d + k0 + 2 * v, acc[0][u][v]); }
        }
        for (long j = 0; j < d; j++) for (long k = j; k < d; k++) if (!NM(covered)(j, k, d)) {
            double s[LN]; for (int l = 0; l < LN; l++) s[l] = 0.0;
            for (long i = 0; i < m; i++) s[i % LN] = SMAC(s[i % LN], xb[i * d + j], xb[i * d + k]);
            for (int c = LN; c > 1; c /= 2) for (int q = 0; q < c / 2; q++) s[q] = s[2 * q] + s[2 * q + 1];
            slot[j * d + k] = s[0];
        }
        sp = settle(st, lv, sp, d * d);
    }
    free(P);
    finish(st, sp, d * d); memcpy(C, st, d * d * 8);
}

/* Blocked right-looking Cholesky, panels of CNB = 32 columns. W: (dp+16) x dp zeroed workspace (dp = d rounded up to 16), the
 * result is the lower triangle of W (leading dimension dp; the upper triangle is scratch). Pt: CNB x (dp+16), the panel transposed. */
#define CNB 32
/* profiling switch only (-DCHOL_NOTRAIL=1 skips the trailing update; outputs are then wrong): used to split the Cholesky time */
#ifndef CHOL_NOTRAIL
#define CHOL_NOTRAIL 0
#endif
static int NM(chol)(const double *A, long d, double *W, double *Pt) {
    const long dp = (d + 15) / 16 * 16, ld = dp, pl = dp + 16;
    for (long i = 0; i < d; i++) { long cnt = (i + 16) / 16 * 16; if (cnt > d) cnt = d; memcpy(W + i * ld, A + i * d, cnt * 8); }
    for (long c0 = 0; c0 < d; c0 += CNB) {
        long c1 = c0 + CNB; if (c1 > d) c1 = d;
        for (long j = c0; j < c1; j++) {
            const long m = j - c0, i0 = j & ~1L; double ljj = 0.0;
            for (long ib = i0; ib < d; ib += 2 * RV) {
                float64x2_t acc[RV][LN];
                UNR for (int v = 0; v < RV; v++) { UNR for (int l = 0; l < LN; l++) acc[v][l] = vdupq_n_f64(0.0); }
                long kk = 0;
                for (; kk + LN <= m; kk += LN) {
                    UNR for (int l = 0; l < LN; l++) {
                        float64x2_t bj = vdupq_n_f64(W[j * ld + c0 + kk + l]); const double *pp = Pt + (kk + l) * pl + ib;
                        UNR for (int v = 0; v < RV; v++) acc[v][l] = VMAC(acc[v][l], vld1q_f64(pp + 2 * v), bj);
                    }
                }
                UNR for (int l = 0; l < LN; l++) if (kk + l < m) {
                    float64x2_t bj = vdupq_n_f64(W[j * ld + c0 + kk + l]); const double *pp = Pt + (kk + l) * pl + ib;
                    UNR for (int v = 0; v < RV; v++) acc[v][l] = VMAC(acc[v][l], vld1q_f64(pp + 2 * v), bj);
                }
                UNR for (int c = LN; c > 1; c /= 2) { UNR for (int q = 0; q < c / 2; q++) { UNR for (int v = 0; v < RV; v++) acc[v][q] = vaddq_f64(acc[v][2 * q], acc[v][2 * q + 1]); } }
                float64x2_t s[RV];
                UNR for (int v = 0; v < RV; v++) {
                    long r0 = ib + 2 * v;
                    float64x2_t av = vcombine_f64(vld1_f64(W + r0 * ld + j), vld1_f64(W + (r0 + 1) * ld + j));
                    s[v] = vsubq_f64(av, acc[v][0]);
                }
                if (ib == i0) {
                    double sj = (j & 1) ? vgetq_lane_f64(s[0], 1) : vgetq_lane_f64(s[0], 0);
                    if (!(sj > 0.0)) return 1;
                    ljj = sqrt(sj);
                }
                float64x2_t dv = vdupq_n_f64(ljj);
                UNR for (int v = 0; v < RV; v++) {
                    long r0 = ib + 2 * v; float64x2_t res = vdivq_f64(s[v], dv);
                    vst1q_f64(Pt + m * pl + r0, res);
                    if (r0 >= j) W[r0 * ld + j] = vgetq_lane_f64(res, 0);
                    W[(r0 + 1) * ld + j] = vgetq_lane_f64(res, 1);
                }
            }
            W[j * ld + j] = ljj; Pt[m * pl + j] = ljj;
        }
        if (c1 < d && !CHOL_NOTRAIL) {  /* trailing update A22 -= L21 L21^T over the 32 panel columns, lane l = (k - c0) mod LN; c1 - c0 == CNB here */
            for (long i0 = c1; i0 < d; i0 += MU) for (long j0 = c1; j0 <= i0 + MU - 1 && j0 < dp; j0 += CW) {
                float64x2_t acc[LN][MU][NV];
                UNR for (int l = 0; l < LN; l++) { UNR for (int u = 0; u < MU; u++) { UNR for (int v = 0; v < NV; v++) acc[l][u][v] = vdupq_n_f64(0.0); } }
                for (long kk = 0; kk < CNB; kk += LN) {
                    UNR for (int l = 0; l < LN; l++) {
                        const double *pp = Pt + (kk + l) * pl; float64x2_t q[NV];
                        UNR for (int v = 0; v < NV; v++) q[v] = vld1q_f64(pp + j0 + 2 * v);
                        UNR for (int u = 0; u < MU; u++) { float64x2_t p = vdupq_n_f64(pp[i0 + u]); UNR for (int v = 0; v < NV; v++) acc[l][u][v] = VMAC(acc[l][u][v], p, q[v]); }
                    }
                }
                UNR for (int c = LN; c > 1; c /= 2) { UNR for (int q = 0; q < c / 2; q++) { UNR for (int u = 0; u < MU; u++) { UNR for (int v = 0; v < NV; v++) acc[q][u][v] = vaddq_f64(acc[2 * q][u][v], acc[2 * q + 1][u][v]); } } }
                UNR for (int u = 0; u < MU; u++) { UNR for (int v = 0; v < NV; v++) { double *p = W + (i0 + u) * ld + j0 + 2 * v; vst1q_f64(p, vsubq_f64(vld1q_f64(p), acc[0][u][v])); } }
            }
        }
    }
    return 0;
}

#undef MU
#undef NV
#undef RV
#undef RW
#undef CW
#undef NR
#undef QL
#undef VMAC
#undef SMAC
#undef CNB
