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

## Stream kinds (`derive(seed, kind)`)

Kind 3 is k-means (`docs/models-linear.md` 3.3): `derive(derive(seed, 3), j)` is the draw of center `j`. Kinds 1 and 2 are the
test-input streams of `scripts/oracle_linalg.py`; kind 3 of those scripts is a separate use inside `linalg-check` (a scalar), not a model stream.
Later models take new numbers; a number is never reused.
