# The random number generator (`lib/core/rng.cho`)

**SplitMix64**, one `int` of state **[choice, design.md section 6]**. Code taken from spike F (`cancho-ml-forest`,
`scripts/spikes/spike_f.cho`), whose vectors are reproduced and checked here.

```
advance(s)    = s + 0x9E3779B97F4A7C15                     (wrapping)
output(s)     = mix(s)                                      mix(z): z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
                                                                  z = (z ^ (z >> 27)) * 0x94D049BB133111EB
                                                                  z ^ (z >> 31)         (all shifts logical, products wrapping)
derive(h, c)  = mix((h ^ c) + 0x9E3779B97F4A7C15)           a child stream: derive(derive(seed, kind), index)
uniform(z, m) = (z >> 1) mod m                              z an output; m <= 0 draws 0
unit(z)       = ((z >> 11) & (2^53 - 1)) * 2^-52 - 1        a float in [-1, 1): exact, no rounding
```

A draw is `s = advance(s); z = output(s)`. A stream is never shared: every tree, fold, sample or test input takes its seed from
`derive`, so no draw depends on which thread runs it (design.md section 6.2).

`uniform` uses the plain modulus, so it has a bias below `m / 2^63`; this is a **choice** (fixed by the spike's vectors, and
irrelevant for the model sizes of v1). `unit` is exact because an integer below 2^53 times a power of two is representable, and
so is the difference with 1.

## Reference vectors (tests/core.cho)

The outputs 1 to 5 for seeds 0, 1234567 and 2^64 - 1 are those of the C reference implementation (Vigna's `next()`,
`cancho-ml-forest/scripts/spikes/splitmix64_ref.c`, compiled and run again for this file: all fifteen agree), and of the numpy
emulation of spike F. Seed 0, first output: `e220a8397b1dcdaf`. `derive(derive(42,1),0) = d8005cdd08a0d146`,
`derive(derive(derive(42,2),3),77) = 28fdfab56efe31a1`, `derive(derive(20261012,1),9) = 533911873c329e77`. The test compares
64-bit values as two 32-bit halves because cancho integer literals are signed. The first `unit` vector (seed 0, output 1) is
`3fe8882a0e5ec772`, from numpy.

The test inputs of the kernels (scripts/oracle_linalg.py) use the same generator: element `i` of a data set of seed `s` is
`unit(output(s + (i + 1) * 0x9E3779B97F4A7C15))` (a stream whose state is advanced `i + 1` times), so numpy can generate a whole
array with vectorised uint64 arithmetic.

## Stream kinds (`derive(seed, kind)`): the registry

A stream is `derive(derive(seed, kind), index)`. **Two uses of one kind would give two components the same numbers for the same seed**
(a k-fold and a forest started from one seed would shuffle with the draws of the bootstrap), so the kinds of the library are one table
and a number is never reused. `lib/core/kinds.cho` has it, `tests/t_registry.cho` fails on a duplicate or on a stage whose own
constant differs, and `scripts/check_registry.py` compares it with this table and with the two places that still write the number.

| kind | stream | index | owner |
|---|---|---|---|
| 1 | bootstrap of tree `t` | `t` | `lib/models/tree/grow.cho` (fixed: spike F's streams, the tree gate compares with them) |
| 2 | features of node `id` of tree `t` | `t`, then `id` (the heap number) | `lib/models/tree/grow.cho` (fixed, as kind 1) |
| 3 | k-means: root of the seeds of the centers | center `j`, then the attempt | `lib/models/kmeans/kmeans.cho` (`docs/models-linear.md` 3.3) |
| 4 | train/test split | the repeat | `lib/metrics/splits.cho` (`docs/metrics.md` 2.1) |
| 5 | k-fold | the repeat | `lib/metrics/splits.cho` |
| 6 | stratified k-fold | `2 * repeat + class` | `lib/metrics/splits.cho` |

The next stream takes 7. *(Renumbered at integration: the metrics stage had 1, 2, 3 for its three splits, which met the tree's 1 and 2
and k-means' 3, and this file had said "kinds 1 and 2 are the test-input streams". The splits' assignment vectors changed with the
numbers; the reference vectors of `docs/metrics.md` were recomputed by its Python oracle, and the oracle and the library agree on all
2,978 cases. The tree's and k-means' numbers did not move, because spike F's identity and the oracle of the models depend on them.)*

**Test inputs are not model streams.** `scripts/oracle_linalg.py`, `tools/linalg-check` and `tools/models-check` build their data with
`derive(seed, 1 .. 3)` where the seed is the generator's own (docs/linalg.md, docs/models-linear.md 6.1); they run in the gate drivers
and never in a library path, so they are outside the table. A library never derives from a test seed.
