
#### dot (unfused, bits identical to the scalar reference of each order)

| shape | CS | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=1000000 | 166.16 us | 227.59 us | 143.14 us | 137.65 us | 134.16 us | N16 | 57.11 us | 278.00 us | 2.35 | 2.91 | 4.87 | NOT MATCH (2.35x) | ok | LOAD>=2 |

#### dot, fma variant (CHANGES BITS; not in the verdict)

| shape | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|
| n=1000000 | 270.75 us | 142.93 us | 135.27 us | 131.26 us | F16 | 2.30 | 0.98 | ok |

#### axpy (unfused, bits identical to the scalar reference of each order)

| shape | C scalar (CS) | C NEON | numpy | cancho (copied) | NEON / numpy | CS / numpy | cancho / numpy | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|
| n=1000000 | 152.71 us | 156.28 us | 73.75 us | 187.10 us | 2.12 | 2.07 | 2.54 | NOT MATCH (2.12x) | LOAD>=2 |

#### gemv_n (unfused, bits identical to the scalar reference of each order)

| shape | CS | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 233.63 us | 158.91 us | 163.88 us | 436.35 us | 746.79 us | N2 | 179.43 us | 313.73 us | 0.89 | 1.30 | 1.75 | MATCH | ok | LOAD>=2 |
| n=100000 d=64 | 777.73 us | 561.96 us | 548.46 us | 546.65 us | 563.28 us | N8 | 325.63 us | 1.067 ms | 1.68 | 2.39 | 3.28 | NOT MATCH (1.68x) | ok | LOAD>=2 |
| n=100000 d=256 | 3.856 ms | 2.689 ms | 2.563 ms | 2.496 ms | 2.512 ms | N8 | 1.652 ms | 5.400 ms | 1.51 | 2.33 | 3.27 | NOT MATCH (1.51x) | ok | LOAD>=2 |
| n=1000000 d=20 | 2.389 ms | 1.794 ms | 1.817 ms | 4.357 ms | 7.553 ms | N2 | 2.121 ms | 3.643 ms | 0.85 | 1.13 | 1.72 | MATCH | ok | LOAD>=2 |
| n=1000000 d=64 | 8.076 ms | 6.288 ms | 6.170 ms | 6.072 ms | 6.165 ms | N8 | 3.836 ms | 13.000 ms | 1.58 | 2.11 | 3.39 | NOT MATCH (1.58x) | ok | LOAD>=2 |
| n=1000000 d=256 | 47.919 ms | 28.157 ms | 27.561 ms | 27.680 ms | 26.532 ms | N16 | 18.028 ms | 0.131 s | 1.47 | 2.66 | 7.27 | NOT MATCH (1.47x) | ok | spread 1.37, LOAD>=2 |

#### gemv_n, fma variant (CHANGES BITS; not in the verdict)

| shape | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 157.47 us | 163.98 us | 431.13 us | 737.69 us | F2 | 0.88 | 0.99 | ok |
| n=100000 d=64 | 536.32 us | 527.83 us | 511.53 us | 535.00 us | F8 | 1.57 | 0.94 | ok |
| n=100000 d=256 | 2.665 ms | 2.461 ms | 2.403 ms | 2.439 ms | F8 | 1.45 | 0.96 | ok |
| n=1000000 d=20 | 1.751 ms | 1.782 ms | 4.300 ms | 7.351 ms | F2 | 0.83 | 0.98 | ok |
| n=1000000 d=64 | 6.136 ms | 5.929 ms | 5.800 ms | 5.933 ms | F8 | 1.51 | 0.96 | ok |
| n=1000000 d=256 | 28.333 ms | 26.524 ms | 26.590 ms | 25.537 ms | F16 | 1.42 | 0.96 | ok |

#### gemv_t (unfused, bits identical to the scalar reference of each order)

| shape | CS | N1 | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 189.51 us | 194.79 us | 185.01 us | 191.97 us | 207.82 us | 217.98 us | N2 | 118.83 us | 296.52 us | 1.56 | 1.59 | 2.50 | NOT MATCH (1.56x) | ok | LOAD>=2 |
| n=100000 d=64 | 583.53 us | 594.63 us | 620.17 us | 624.86 us | 812.71 us | 847.20 us | N1 | 319.29 us | 900.00 us | 1.86 | 1.83 | 2.82 | NOT MATCH (1.86x) | ok | LOAD>=2 |
| n=100000 d=256 | 2.649 ms | 2.682 ms | 3.161 ms | 3.216 ms | 3.377 ms | 3.345 ms | N1 | 1.252 ms | 4.520 ms | 2.14 | 2.12 | 3.61 | NOT MATCH (2.14x) | ok | LOAD>=2 |
| n=1000000 d=20 | 2.003 ms | 2.049 ms | 1.983 ms | 1.998 ms | 2.176 ms | 2.556 ms | N2 | 1.389 ms | 3.143 ms | 1.43 | 1.44 | 2.26 | NOT MATCH (1.43x) | ok | LOAD>=2 |
| n=1000000 d=64 | 6.334 ms | 6.452 ms | 6.610 ms | 6.815 ms | 8.228 ms | 8.511 ms | N1 | 3.281 ms | 11.667 ms | 1.97 | 1.93 | 3.56 | NOT MATCH (1.97x) | ok | LOAD>=2 |
| n=1000000 d=256 | 27.650 ms | 30.776 ms | 32.001 ms | 32.935 ms | 34.138 ms | 34.418 ms | N1 | 12.903 ms | 95.000 ms | 2.39 | 2.14 | 7.36 | NOT MATCH (2.39x) | ok | LOAD>=2 |

#### gemv_t, fma variant (CHANGES BITS; not in the verdict)

| shape | F1 | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 204.67 us | 184.86 us | 190.03 us | 203.54 us | 203.75 us | F2 | 1.56 | 1.00 | ok |
| n=100000 d=64 | 586.68 us | 605.89 us | 609.93 us | 812.42 us | 835.83 us | F1 | 1.84 | 0.99 | ok |
| n=100000 d=256 | 2.537 ms | 3.096 ms | 3.210 ms | 3.337 ms | 3.322 ms | F1 | 2.03 | 0.95 | ok |
| n=1000000 d=20 | 2.120 ms | 1.963 ms | 1.963 ms | 2.142 ms | 2.431 ms | F4 | 1.41 | 0.99 | ok |
| n=1000000 d=64 | 6.407 ms | 6.447 ms | 6.749 ms | 8.140 ms | 8.383 ms | F1 | 1.95 | 0.99 | ok |
| n=1000000 d=256 | 27.005 ms | 31.629 ms | 32.119 ms | 34.040 ms | 34.180 ms | F1 | 2.09 | 0.88 | ok |

#### syrk (unfused, bits identical to the scalar reference of each order)

| shape | CS | N1 | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 1.593 ms | 4.234 ms | 1.661 ms | 1.795 ms | 1.499 ms | 2.095 ms | N8 | 525.82 us | 1.880 ms | 2.85 | 3.03 | 3.58 | NOT MATCH (2.85x) | ok | LOAD>=2 |
| n=100000 d=64 | 14.924 ms | 17.471 ms | 17.368 ms | 18.278 ms | 16.863 ms | 23.679 ms | N8 | 1.486 ms | 17.375 ms | 11.34 | 10.04 | 11.69 | NOT MATCH (11.34x) | ok | LOAD>=2 |
| n=100000 d=256 | 0.269 s | 0.225 s | 0.234 s | 0.257 s | 0.264 s | 0.423 s | N1 | 17.762 ms | 0.290 s | 12.68 | 15.14 | 16.33 | NOT MATCH (12.68x) | - | LOAD>=2 |
| n=1000000 d=20 | 18.247 ms | 44.127 ms | 20.741 ms | 20.579 ms | 17.570 ms | 23.565 ms | N8 | 5.540 ms | 22.333 ms | 3.17 | 3.29 | 4.03 | NOT MATCH (3.17x) | ok | LOAD>=2 |
| n=1000000 d=64 | 0.159 s | 0.196 s | 0.195 s | 0.203 s | 0.186 s | 0.259 s | N8 | 14.989 ms | 0.180 s | 12.44 | 10.63 | 12.01 | NOT MATCH (12.44x) | - | LOAD>=2 |
| n=1000000 d=256 | 2.840 s | 2.423 s | 2.463 s | 2.707 s | 2.747 s | 4.437 s | N1 | 0.178 s | 2.734 s | 13.60 | 15.95 | 15.35 | NOT MATCH (13.60x) | - | LOAD>=2 |

#### syrk, fma variant (CHANGES BITS; not in the verdict)

| shape | F1 | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 4.516 ms | 1.013 ms | 1.234 ms | 1.202 ms | 2.067 ms | F2 | 1.93 | 0.68 | ok |
| n=100000 d=64 | 11.134 ms | 11.035 ms | 13.689 ms | 13.900 ms | 23.036 ms | F2 | 7.42 | 0.65 | ok |
| n=100000 d=256 | 0.134 s | 0.147 s | 0.210 s | 0.227 s | 0.414 s | F1 | 7.57 | 0.60 | - |
| n=1000000 d=20 | 47.382 ms | 12.812 ms | 15.626 ms | 14.506 ms | 22.895 ms | F2 | 2.31 | 0.73 | ok |
| n=1000000 d=64 | 0.133 s | 0.131 s | 0.157 s | 0.159 s | 0.251 s | F2 | 8.75 | 0.70 | - |
| n=1000000 d=256 | 1.417 s | 1.510 s | 2.218 s | 2.397 s | 4.386 s | F1 | 7.96 | 0.58 | - |

#### chol (unfused, bits identical to the scalar reference of each order)

| shape | CS | N1 | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d=21 | 0.35 us | 0.40 us | 0.40 us | 0.34 us | 0.38 us | 0.57 us | N4 | 2.38 us | 0.58 us | 0.14 | 0.15 | 0.24 | MATCH | ok | LOAD>=2; vs raw dpotrf 0.49 us: 0.70x |
| d=64 | 7.23 us | 4.55 us | 4.36 us | 4.36 us | 4.65 us | 8.24 us | N2 | 8.90 us | 8.76 us | 0.49 | 0.81 | 0.98 | MATCH | ok | LOAD>=2; vs raw dpotrf 5.20 us: 0.84x |
| d=256 | 481.81 us | 246.79 us | 245.14 us | 274.92 us | 264.37 us | 490.69 us | N2 | 146.76 us | 490.26 us | 1.67 | 3.28 | 3.34 | NOT MATCH (1.67x) | ok | LOAD>=2; vs raw dpotrf 47.25 us: 5.19x |

#### chol, fma variant (CHANGES BITS; not in the verdict)

| shape | F1 | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|---|
| d=21 | 0.34 us | 0.35 us | 0.31 us | 0.36 us | 0.55 us | F4 | 0.13 | 0.90 | ok |
| d=64 | 3.38 us | 3.44 us | 3.87 us | 4.14 us | 7.70 us | F1 | 0.38 | 0.78 | ok |
| d=256 | 171.06 us | 185.64 us | 296.13 us | 253.08 us | 468.71 us | F1 | 1.17 | 0.70 | ok |

#### Bandwidth of the memory-bound cells (bytes of X read / time; dot counts both vectors, axpy counts x read, y read and written)

| kernel | shape | bytes | N(best) GB/s | CS GB/s | numpy GB/s | cancho GB/s (copied) | best read probe GB/s (size class) | time at probe rate / numpy |
|---|---|---|---|---|---|---|---|---|
| dot | n=1000000 | 16.0 MB | 119.3 (N16) | 96.3 | 280.2 | 57.6 | 121.2 (N=2000000) | 2.31 |
| axpy | n=1000000 | 24.0 MB | 153.6 (N) | 157.2 | 325.4 | 128.3 | 121.2 (N=2000000) | 2.68 |
| gemv_n | n=100000 d=20 | 16.0 MB | 100.7 (N2) | 68.5 | 89.2 | 51.0 | 121.2 (N=2000000) | 0.74 |
| gemv_n | n=100000 d=64 | 51.2 MB | 93.7 (N8) | 65.8 | 157.2 | 48.0 | 118.8 (N=6400000) | 1.32 |
| gemv_n | n=100000 d=256 | 204.8 MB | 82.1 (N8) | 53.1 | 123.9 | 37.9 | 114.1 (N=25600000) | 1.09 |
| gemv_n | n=1000000 d=20 | 160.0 MB | 89.2 (N2) | 67.0 | 75.4 | 43.9 | 114.1 (N=25600000) | 0.66 |
| gemv_n | n=1000000 d=64 | 512.0 MB | 84.3 (N8) | 63.4 | 133.5 | 39.4 | 114.1 (N=25600000) | 1.17 |
| gemv_n | n=1000000 d=256 | 2048.0 MB | 77.2 (N16) | 42.7 | 113.6 | 15.6 | 113.5 (N=256000000) | 1.00 |
| gemv_t | n=100000 d=20 | 16.0 MB | 86.5 (N2) | 84.4 | 134.6 | 54.0 | 121.2 (N=2000000) | 1.11 |
| gemv_t | n=100000 d=64 | 51.2 MB | 86.1 (N1) | 87.7 | 160.4 | 56.9 | 118.8 (N=6400000) | 1.35 |
| gemv_t | n=100000 d=256 | 204.8 MB | 76.4 (N1) | 77.3 | 163.5 | 45.3 | 114.1 (N=25600000) | 1.43 |
| gemv_t | n=1000000 d=20 | 160.0 MB | 80.7 (N2) | 79.9 | 115.2 | 50.9 | 114.1 (N=25600000) | 1.01 |
| gemv_t | n=1000000 d=64 | 512.0 MB | 79.4 (N1) | 80.8 | 156.0 | 43.9 | 114.1 (N=25600000) | 1.37 |
| gemv_t | n=1000000 d=256 | 2048.0 MB | 66.5 (N1) | 74.1 | 158.7 | 21.6 | 113.5 (N=256000000) | 1.40 |

#### What the fixed order costs on dot: N(L) with blocks and a lane tree per block vs 16 free lanes with no blocks

* dot n=1000000: free16 128.61 us, N(best=N16) 134.16 us, ratio fixed/free = 1.04; N by L: N2 227.59 us, N4 143.14 us, N8 137.65 us, N16 134.16 us

#### Spreads and load per cell

| kernel | shape | spread numpy | spread CS | spread N(best) | load start / end | attempts |
|---|---|---|---|---|---|---|
| dot | n=1000000 | 1.18 | 1.07 | 1.03 | 2.3 / 2.3 | 2 |
| axpy | n=1000000 | 1.05 | 1.05 | 1.08 | 2.3 / 2.3 | 1 |
| gemv_n | n=100000 d=20 | 1.02 | 1.07 | 1.04 | 2.3 / 2.3 | 1 |
| gemv_n | n=100000 d=64 | 1.23 | 1.14 | 1.04 | 2.3 / 2.2 | 1 |
| gemv_n | n=100000 d=256 | 1.15 | 1.10 | 1.06 | 2.4 / 2.4 | 3 |
| gemv_n | n=1000000 d=20 | 1.04 | 1.03 | 1.01 | 2.3 / 2.3 | 1 |
| gemv_n | n=1000000 d=64 | 1.09 | 1.01 | 1.04 | 2.4 / 2.4 | 1 |
| gemv_n | n=1000000 d=256 | 1.08 | 1.37 | 1.09 | 3.2 / 3.2 | 2 |
| gemv_t | n=100000 d=20 | 1.17 | 1.02 | 1.07 | 2.3 / 2.3 | 1 |
| gemv_t | n=100000 d=64 | 1.04 | 1.04 | 1.06 | 2.2 / 2.2 | 1 |
| gemv_t | n=100000 d=256 | 1.01 | 1.05 | 1.12 | 2.4 / 2.4 | 1 |
| gemv_t | n=1000000 d=20 | 1.01 | 1.12 | 1.08 | 2.3 / 2.3 | 1 |
| gemv_t | n=1000000 d=64 | 1.01 | 1.05 | 1.12 | 2.4 / 2.3 | 1 |
| gemv_t | n=1000000 d=256 | 1.07 | 1.05 | 1.15 | 3.1 / 3.0 | 1 |
| syrk | n=100000 d=20 | 1.02 | 1.09 | 1.11 | 2.3 / 2.3 | 1 |
| syrk | n=100000 d=64 | 1.05 | 1.06 | 1.04 | 2.2 / 2.4 | 1 |
| syrk | n=100000 d=256 | 1.01 | 1.05 | 1.06 | 2.4 / 2.3 | 1 |
| syrk | n=1000000 d=20 | 1.04 | 1.05 | 1.05 | 2.3 / 2.2 | 1 |
| syrk | n=1000000 d=64 | 1.01 | 1.06 | 1.06 | 3.0 / 3.2 | 3 |
| syrk | n=1000000 d=256 | 1.05 | 1.15 | 1.02 | 6.1 / 6.3 | 2 |
| chol | d=21 | 1.16 | 1.10 | 1.03 | 6.3 / 5.5 | 1 |
| chol | d=64 | 1.03 | 1.07 | 1.06 | 5.5 / 5.3 | 1 |
| chol | d=256 | 1.08 | 1.05 | 1.05 | 5.3 / 5.3 | 1 |

#### Compute-bound cells against the measured register-only peak (scripts/simd/peak.c: unfused 14.9 GMAC/s, fma 23.9 GMAC/s)

| kernel | shape | useful MACs | N(best) | GMAC/s | % of unfused peak | F(best) | GMAC/s | % of fma peak | CS GMAC/s |
|---|---|---|---|---|---|---|---|---|---|
| syrk | n=100000 d=20 | 2.1e+07 | N8 | 14.01 | 94% | F2 | 20.72 | 87% | 13.18 |
| syrk | n=100000 d=64 | 2.08e+08 | N8 | 12.33 | 83% | F2 | 18.85 | 79% | 13.94 |
| syrk | n=100000 d=256 | 3.29e+09 | N1 | 14.61 | 98% | F1 | 24.47 | 102% | 12.23 |
| syrk | n=1000000 d=20 | 2.1e+08 | N8 | 11.95 | 80% | F2 | 16.39 | 68% | 11.51 |
| syrk | n=1000000 d=64 | 2.08e+09 | N8 | 11.16 | 75% | F2 | 15.86 | 66% | 13.06 |
| syrk | n=1000000 d=256 | 3.29e+10 | N1 | 13.58 | 91% | F1 | 23.21 | 97% | 11.58 |
| chol | d=21 | 1.54e+03 | N4 | 4.49 | 30% | F4 | 4.98 | 21% | 4.37 |
| chol | d=64 | 4.37e+04 | N2 | 10.03 | 67% | F1 | 12.91 | 54% | 6.05 |
| chol | d=256 | 2.8e+06 | N2 | 11.41 | 76% | F1 | 16.35 | 68% | 5.80 |

#### Verdict table: would a SIMD cancho at this ceiling MATCH numpy? (unfused, bit-identical to the scalar reference; fma shown apart)

| kernel | shape | numpy | N(best L) | N / numpy | verdict | min(CS,N) / numpy | F(best) / numpy (bits change) | cancho (copied) / N(best) | notes |
|---|---|---|---|---|---|---|---|---|---|
| dot | n=1000000 | 57.11 us | N16 134.16 us | 2.35 | NOT MATCH | 2.35 | 2.30 | 2.07x | LOAD>=2 |
| axpy | n=1000000 | 73.75 us | N 156.28 us | 2.12 | NOT MATCH | 2.07 | 2.08 | 1.20x | LOAD>=2 |
| gemv_n | n=100000 d=20 | 179.43 us | N2 158.91 us | 0.89 | MATCH | 0.89 | 0.88 | 1.97x | LOAD>=2 |
| gemv_n | n=100000 d=64 | 325.63 us | N8 546.65 us | 1.68 | NOT MATCH | 1.68 | 1.57 | 1.95x | LOAD>=2 |
| gemv_n | n=100000 d=256 | 1.652 ms | N8 2.496 ms | 1.51 | NOT MATCH | 1.51 | 1.45 | 2.16x | LOAD>=2 |
| gemv_n | n=1000000 d=20 | 2.121 ms | N2 1.794 ms | 0.85 | MATCH | 0.85 | 0.83 | 2.03x | LOAD>=2 |
| gemv_n | n=1000000 d=64 | 3.836 ms | N8 6.072 ms | 1.58 | NOT MATCH | 1.58 | 1.51 | 2.14x | LOAD>=2 |
| gemv_n | n=1000000 d=256 | 18.028 ms | N16 26.532 ms | 1.47 | NOT MATCH | 1.47 | 1.42 | 4.94x | spread 1.37, LOAD>=2 |
| gemv_t | n=100000 d=20 | 118.83 us | N2 185.01 us | 1.56 | NOT MATCH | 1.56 | 1.56 | 1.60x | LOAD>=2 |
| gemv_t | n=100000 d=64 | 319.29 us | N1 594.63 us | 1.86 | NOT MATCH | 1.83 | 1.84 | 1.51x | LOAD>=2 |
| gemv_t | n=100000 d=256 | 1.252 ms | N1 2.682 ms | 2.14 | NOT MATCH | 2.12 | 2.03 | 1.69x | LOAD>=2 |
| gemv_t | n=1000000 d=20 | 1.389 ms | N2 1.983 ms | 1.43 | NOT MATCH | 1.43 | 1.41 | 1.59x | LOAD>=2 |
| gemv_t | n=1000000 d=64 | 3.281 ms | N1 6.452 ms | 1.97 | NOT MATCH | 1.93 | 1.95 | 1.81x | LOAD>=2 |
| gemv_t | n=1000000 d=256 | 12.903 ms | N1 30.776 ms | 2.39 | NOT MATCH | 2.14 | 2.09 | 3.09x | LOAD>=2 |
| syrk | n=100000 d=20 | 525.82 us | N8 1.499 ms | 2.85 | NOT MATCH | 2.85 | 1.93 | 1.25x | LOAD>=2 |
| syrk | n=100000 d=64 | 1.486 ms | N8 16.863 ms | 11.34 | NOT MATCH | 10.04 | 7.42 | 1.03x | LOAD>=2 |
| syrk | n=100000 d=256 | 17.762 ms | N1 0.225 s | 12.68 | NOT MATCH | 12.68 | 7.57 | 1.29x | LOAD>=2 |
| syrk | n=1000000 d=20 | 5.540 ms | N8 17.570 ms | 3.17 | NOT MATCH | 3.17 | 2.31 | 1.27x | LOAD>=2 |
| syrk | n=1000000 d=64 | 14.989 ms | N8 0.186 s | 12.44 | NOT MATCH | 10.63 | 8.75 | 0.97x | LOAD>=2 |
| syrk | n=1000000 d=256 | 0.178 s | N1 2.423 s | 13.60 | NOT MATCH | 13.60 | 7.96 | 1.13x | LOAD>=2 |
| chol | d=21 | 2.38 us | N4 0.34 us | 0.14 | MATCH | 0.14 | 0.13 | 1.69x | LOAD>=2; vs raw dpotrf: 0.70 |
| chol | d=64 | 8.90 us | N2 4.36 us | 0.49 | MATCH | 0.49 | 0.38 | 2.01x | LOAD>=2; vs raw dpotrf: 0.84 |
| chol | d=256 | 146.76 us | N2 245.14 us | 1.67 | NOT MATCH | 1.67 | 1.17 | 2.00x | LOAD>=2; vs raw dpotrf: 5.19 |

#### Which L: time of N(L) over the time of the best L, per cell (1.00 = best); worst case and geometric mean over the cells of each kernel

| kernel | L=1 worst / geomean | L=2 worst / geomean | L=4 worst / geomean | L=8 worst / geomean | L=16 worst / geomean |
|---|---|---|---|---|---|
| dot | n/a | 1.70 / 1.70 | 1.07 / 1.07 | 1.03 / 1.03 | 1.00 / 1.00 |
| gemv_n | n/a | 1.08 / 1.03 | 1.04 / 1.02 | 2.75 / 1.38 | 4.70 / 1.66 |
| gemv_t | 1.05 / 1.01 | 1.18 / 1.05 | 1.20 / 1.07 | 1.37 / 1.20 | 1.42 / 1.26 |
| syrk | 2.82 / 1.41 | 1.18 / 1.07 | 1.20 / 1.13 | 1.17 / 1.05 | 1.88 / 1.52 |
| chol | 1.17 / 1.07 | 1.17 / 1.05 | 1.12 / 1.04 | 1.11 / 1.09 | 2.00 / 1.85 |
| all cells where L exists for every kernel (dot, gemv_n, gemv_t, syrk, chol) | n/a (no dot, gemv_n) | 1.70 / 1.07 | 1.20 / 1.07 | 2.75 / 1.18 | 4.70 / 1.49 |

#### The lane order against the current cancho order, vectorised or not: CS (current order, plain C) / N(best L) and N(1) (current order for gemv_t, syrk, vectorised across outputs) / N(best L)

| kernel | shape | CS / N(best) | N1 / N(best) |
|---|---|---|---|
| dot | n=1000000 | 1.24 | - |
| gemv_n | n=100000 d=20 | 1.47 | - |
| gemv_n | n=100000 d=64 | 1.42 | - |
| gemv_n | n=100000 d=256 | 1.55 | - |
| gemv_n | n=1000000 d=20 | 1.33 | - |
| gemv_n | n=1000000 d=64 | 1.33 | - |
| gemv_n | n=1000000 d=256 | 1.81 | - |
| gemv_t | n=100000 d=20 | 1.02 | 1.05 |
| gemv_t | n=100000 d=64 | 0.98 | 1.00 |
| gemv_t | n=100000 d=256 | 0.99 | 1.00 |
| gemv_t | n=1000000 d=20 | 1.01 | 1.03 |
| gemv_t | n=1000000 d=64 | 0.98 | 1.00 |
| gemv_t | n=1000000 d=256 | 0.90 | 1.00 |
| syrk | n=100000 d=20 | 1.06 | 2.82 |
| syrk | n=100000 d=64 | 0.89 | 1.04 |
| syrk | n=100000 d=256 | 1.19 | 1.00 |
| syrk | n=1000000 d=20 | 1.04 | 2.51 |
| syrk | n=1000000 d=64 | 0.85 | 1.05 |
| syrk | n=1000000 d=256 | 1.17 | 1.00 |
| chol | d=21 | 1.03 | 1.17 |
| chol | d=64 | 1.66 | 1.04 |
| chol | d=256 | 1.97 | 1.01 |
