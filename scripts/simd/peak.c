/* Register-only vector throughput of one core (no loads, no stores; 16 independent chains, operands from a volatile so nothing is
 * hoisted): vector adds, vector multiplies, vector fma, and the unfused multiply-add pair (vmulq + vaddq) as a kernel uses it.
 * One vector op = 2 lanes. Prints vector ops per ns; at the clock (about 4.4 GHz here, not read from the machine) divide for per cycle.
 *     cc -O3 -mcpu=native -ffp-contract=off -o build/peak scripts/simd/peak.c ; build/peak */
#include <arm_neon.h>
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static volatile double cv = 1e-30;
#define BODY(EXPR) __attribute__((noinline)) static double run_##EXPR##_(long it) { float64x2_t a[16]; float64x2_t c = vdupq_n_f64(cv); \
    for (int i = 0; i < 16; i++) a[i] = vdupq_n_f64(1.0 + i * 1e-3); \
    for (long k = 0; k < it; k++) for (int i = 0; i < 16; i++) { float64x2_t o = a[(i + 1) & 15]; (void)o; a[i] = EXPR_##EXPR; } \
    float64x2_t s = a[0]; for (int i = 1; i < 16; i++) s = vaddq_f64(s, a[i]); return vaddvq_f64(s); }
#define EXPR_add vaddq_f64(a[i], o)
#define EXPR_mul vmulq_f64(a[i], vaddq_f64(c, c))
#define EXPR_fma vfmaq_f64(a[i], o, c)
#define EXPR_pair vaddq_f64(a[i], vmulq_f64(o, c))
BODY(add) BODY(mul) BODY(fma) BODY(pair)
int main(void) {
    volatile double sink = 0; long it = 20000000;
    struct { const char *n; double (*f)(long); double ops; } T[] = {{"vadd (1 op)", run_add_, 1}, {"vmul (1 op, +1 add of consts, see note)", run_mul_, 1}, {"vfma (1 op)", run_fma_, 1}, {"unfused pair vmul+vadd (2 ops)", run_pair_, 2}};
    for (int t = 0; t < 4; t++) { double best = 1e9; for (int r = 0; r < 7; r++) { double q = now(); sink += T[t].f(it); q = now() - q; if (q < best) best = q; }
        printf("%-42s %6.2f G chains-steps/s  = %6.2f G vector ops/s\n", T[t].n, 16.0 * it / best / 1e9, 16.0 * it / best * T[t].ops / 1e9); }
    return 0; }
