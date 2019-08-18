[![Build Status](https://travis-ci.com/mklarqvist/libalgebra.svg)](https://travis-ci.com/mklarqvist/libalgebra)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/mklarqvist/libalgebra?branch=master&svg=true)](https://ci.appveyor.com/project/mklarqvist/libalgebra)
[![Github Releases](https://img.shields.io/github/release/mklarqvist/libalgebra.svg)](https://github.com/mklarqvist/libalgebra/releases)
[![License](https://img.shields.io/badge/Apache-2.0-blue.svg)](LICENSE)

# libalgebra

```libalgebra.h``` is a header-only C/C++ library for:
* counting the number of set bits ("population count", `popcnt`) in an array
* counting the number of set bits at each position ("positional population count", `pospopcnt`) in an array
* perform set algebraic operations on bitmaps including union, intersection, and diff cardinalities

using specialized CPU instructions i.e.
[POPCNT](https://en.wikipedia.org/wiki/SSE4#POPCNT_and_LZCNT),
[SSE4.2](https://en.wikipedia.org/wiki/SSE4#SSE4.2),
[AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions),
[AVX512BW](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions),
[NEON](https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_.28NEON.29). ```libalgebra.h``` has been tested successfully using the GCC,
Clang and MSVC compilers.

The core algorithms are described in the papers:

* [Faster Population Counts using AVX2 Instructions](https://arxiv.org/abs/1611.07612) by Daniel Lemire, Nathan Kurz
  and Wojciech Muła (23 Nov 2016).
* Efficient Computation of Positional Population Counts Using SIMD Instructions,
  by Marcus D. R. Klarqvist, Wojciech Muła, and Daniel Lemire (upcoming)
* [Consistently faster and smaller compressed bitmaps with Roaring](https://arxiv.org/abs/1603.06549) by D. Lemire, G. Ssi-Yan-Kai,
  and O. Kaser (21 Mar 2016).

### Speedup

Sample performance metrics (practical upper limit) on AVX512BW machine. We simulate a single data array or pairs of data arrays in a aligned memory location and compute the same statistics many times using the command `benchmark -p -r 10000` (required Linux `perf` subsystem). This reflect the fastest possible throughput if you never have to leave the destination cache-level.
The host architecture used is a 10 nm Cannon Lake [Core i3-8121U](https://ark.intel.com/content/www/us/en/ark/products/136863/intel-core-i3-8121u-processor-4m-cache-up-to-3-20-ghz.html) with gcc (GCC) 8.2.1 20180905 (Red Hat 8.2.1-3).

### POSPOPCNT

This benchmark shows the speedup of the four `pospopcnt` algorithms used on x86
CPUs compared to a naive unvectorized solution
(`pospopcnt_u16_scalar_naive_nosimd`) for different array sizes (in number of
2-byte values). 

| Algorithm                         | 128  | 256   | 512   | 1024  | 2048  | 4096  | 8192  | 65536  |
|-----------------------------------|------|-------|-------|-------|-------|-------|-------|--------|
| pospopcnt_u16_sse_blend_popcnt_unroll8    | **8.28** | 9.84  | 10.55 | 11    | 11.58 | 11.93 | 12.13 | 12.28  |
| pospopcnt_u16_avx512_blend_popcnt_unroll8 | 7.07 | **11.25** | **16.21** | 21    | 25.49 | 27.91 | 29.73 | 31.55  |
| pospopcnt_u16_avx512_adder_forest        | 3.05 | 2.82  | 14.53 | **23.13** | **34.37** | 44.91 | 52.78 | 61.68  |
| pospopcnt_u16_avx512_harvey_seal          | 2.07 | 2.3   | 8.21  | 15.41 | 28.17 | **49.14** | **76.11** | **138.71** |

### POPCNT

Fold speedup compared to a naive unvectorized algorithm
(`popcount_scalar_naive_nosimd`) for different array sizes as (CPU cycles/64-bit word, Instructions/64-bit word):

| Words   | libalgebra.h | Scalar        | Speedup |
|---------|--------------|---------------|---------|
| 4       | 27.75 (37)   | 26.75 (33.5)  | 1       |
| 8       | 16.38 (25.5) | 17.38 (30.25) | 1.1     |
| 16      | 10.5 (19.94) | 12.75 (28.63) | 1.2     |
| 32      | 7.72 (17.16) | 10.69 (27.81) | 1.4     |
| 64      | 3.09 (4.36)  | 9.61 (27.41)  | 3.1     |
| 128     | 2.53 (2.73)  | 8.84 (27.2)   | 3.5     |
| 256     | 1.35 (1.7)   | 8.5 (27.1)    | 6.3     |
| 512     | 0.67 (1.18)  | 8.33 (27.05)  | 12.4    |
| 1024    | 0.5 (0.92)   | 8.25 (27.03)  | 16.4    |
| 2048    | 0.41 (0.79)  | 8.15 (27.01)  | 20.1    |
| 4096    | 0.46 (0.72)  | 8.12 (27.01)  | 17.8    |
| 8192    | 0.39 (0.69)  | 8.11 (27)     | 21      |
| 16384   | 0.39 (0.67)  | 8.1 (27)      | 20.6    |
| 32768   | 0.89 (0.66)  | 8.1 (27)      | 9.1     |
| 65536   | 0.84 (0.66)  | 8.1 (27)      | 9.6     |
| 131072  | 0.68 (0.66)  | 8.09 (27)     | 11.9    |
| 262144  | 1.11 (0.66)  | 8.09 (27)     | 7.3     |
| 524288  | 1.84 (0.66)  | 8.12 (27)     | 4.4     |
| 1048576 | 1.95 (0.66)  | 8.15 (27)     | 4.2     |

### Set algebra

Fold speedup compared to naive unvectorized solution (`*_scalar_naive_nosimd`)
for different array sizes (in number of _pairs_ of 64-bit word but results reported per _single_ 64-bit word). These
functions are identifical with the exception of the bitwise operator used (AND,
OR, or XOR) which all have identical latency and throughput (CPI).

| Words   | libalgebra.h | Scalar        | Speedup |
|---------|--------------|---------------|---------|
| 4       | 17.63 (8.63) | 14.63 (22.75) | 0.8     |
| 8       | 8.13 (5.44)  | 10 (20.88)    | 1.2     |
| 16      | 4.69 (3.84)  | 7.91 (19.94)  | 1.7     |
| 32      | 2.38 (2.56)  | 6.59 (19.47)  | 2.8     |
| 64      | 1.82 (2.06)  | 5.87 (19.23)  | 3.2     |
| 128     | 0.88 (0.89)  | 5.43 (19.12)  | 6.2     |
| 256     | 0.57 (0.64)  | 5.18 (19.06)  | 9.2     |
| 512     | 0.41 (0.51)  | 5.11 (19.03)  | 12.4    |
| 1024    | 0.33 (0.45)  | 5.06 (19.02)  | 15.3    |
| 2048    | 0.39 (0.41)  | 5.03 (19.01)  | 13.1    |
| 4096    | 0.36 (0.4)   | 5.02 (19)     | 13.9    |
| 8192    | 0.37 (0.39)  | 5.01 (19)     | 13.7    |
| 16384   | 0.55 (0.39)  | 5.01 (19)     | 9.1     |
| 32768   | 0.55 (0.39)  | 5 (19)        | 9.2     |
| 65536   | 0.52 (0.38)  | 5 (19)        | 9.7     |
| 131072  | 0.56 (0.38)  | 5.01 (19)     | 9       |
| 262144  | 1.25 (0.38)  | 5.02 (19)     | 4       |
| 524288  | 1.76 (0.38)  | 5.03 (19)     | 2.9     |
| 1048576 | 1.81 (0.38)  | 5.07 (19)     | 2.8     |

## C/C++ API

```C
#include "libalgebra.h"

/*
 * Count the number of 1 bits in the data array
 * @data: An array
 * @size: Size of data in bytes
 */
uint64_t STORM_popcnt(const void* data, uint64_t size);
```

```C
#include "libalgebra.h"

/*
 * Count the number of 1 bits for each position in the data array
 * @data: A 16-bit array
 * @size: Size of data in bytes
 * @flags: Output vector[16]
 */
uint32_t flags[16];
int STORM_pospopcnt_u16(const uint16_t* data, uint32_t size, uint32_t* flags);
```

```C
#include "libalgebra.h"

/*
 * Compute the intersection, union, or diff cardinality between pairs of bitmaps
 * @data1: A 64-bit array
 * @data2: A 64-bit array
 * @size: Size of data in 64-bit words
 */
// Intersect cardinality
uint64_t STORM_intersect_count(const uint64_t* data1, const uint64_t* data2, const uint32_t size);
// Union cardinality
uint64_t STORM_union_count(const uint64_t* data1, const uint64_t* data2, const uint32_t size);
// Diff cardinality
uint64_t STORM_diff_count(const uint64_t* data1, const uint64_t* data2, const uint32_t size);
```

### Advanced use

Retrieve a function pointer to the optimal function given the target length.

```C
STORM_compute_func STORM_get_intersection_count_func(const size_t n_bitmaps_vector);
STORM_compute_func STORM_get_union_count_func(const size_t n_bitmaps_vector);
STORM_compute_func STORM_get_diff_count_func(const size_t n_bitmaps_vector);
```

Portable memory alignment.

```C
#include "libalgebra.h"

void* STORM_aligned_malloc(size_t alignment, size_t size);
void STORM_aligned_free(void* memblock);
```

## How it works

On x86 CPUs ```libalgebra.h``` uses a combination of algorithms depending on the input vector size and what instruction set your CPU supports. These checks are performed during **run-time**.