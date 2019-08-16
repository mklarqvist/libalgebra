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

Sample performance metrics (practical upper limit) on AVX512BW machine. We simulate a single data array or pairs of data arrays and iterate 
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

stuff

### Set algebra

Fold speedup compared to naive unvectorized solution
(`*_scalar_naive_nosimd`) for different array sizes (in number of _pairs_ of 64-bit values).

| Algorithm       | 8    | 32    | 128   | 256   | 512   | 1024  | 2048  | 4096  | 8192  |
|-----------------|------|-------|-------|-------|-------|-------|-------|-------|-------|
| intersect count | 4.73 | 10.8  | 17.58 | 24.82 | 31    | 35.78 | 37.75 | 23.08 | 20.81 |
| union count     | 4.64 | 10.96 | 17.19 | 24.88 | 31.09 | 35.74 | 37.95 | 22.92 | 21.11 |
| diff count      | 4.57 | 10.93 | 17.31 | 24.78 | 30.98 | 35.74 | 37.87 | 23.31 | 21.42 |

Same table showing throughput as MB/s and CPU cycles / 8-byte word:

| Algorithm       | 8              | 32             | 128            | 256           | 512           | 1024          | 2048          | 4096         | 8192          |
|-----------------|----------------|----------------|----------------|---------------|---------------|---------------|---------------|--------------|---------------|
| intersect count | 24414.1 (1.61) | 48828.1 (0.71) | 84918.5 (0.41) | 122070 (0.28) | 150240 (0.22) | 173611 (0.19) | 183824 (0.18) | 112007 (0.3) | 100644 (0.33) |
| union count     | 24414.1 (1.57) | 48828.1 (0.72) | 84918.5 (0.41) | 122070 (0.28) | 150240 (0.22) | 173611 (0.19) | 184911 (0.18) | 111210 (0.3) | 102124 (0.33) |
| diff count      | 24414.1 (1.57) | 48828.1 (0.72) | 84918.5 (0.41) | 122070 (0.28) | 150240 (0.22) | 173611 (0.19) | 183824 (0.18) | 113020 (0.3) | 103648 (0.32) |

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
 * @size: Size of data in bytes
 */
// Intersect cardinality
uint64_t STORM_intersect_count(const uint64_t* data1, const uint64_t* data2, const uint32_t size);
// Union cardinality
uint64_t STORM_union_count(const uint64_t* data1, const uint64_t* data2, const uint32_t size);
// Diff cardinality
uint64_t STORM_diff_count(const uint64_t* data1, const uint64_t* data2, const uint32_t size);
```

## How it works

On x86 CPUs ```libalgebra.h``` uses a combination of algorithms depending on the input vector size and what instruction set your CPU supports. These checks are performed during **run-time**.