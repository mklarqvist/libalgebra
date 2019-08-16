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
[NEON](https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_.28NEON.29).
```libalgebra.h``` has been tested successfully using the GCC,
Clang and MSVC compilers.

### Speedup

### POSPOPCNT

This benchmark shows the speedup of the 3four `pospopcnt` algorithms used on x86 CPUs compared to a naive unvectorized solution (`pospopcnt_u16_scalar_naive_nosimd`) for different array sizes (in number of 2-byte values). 

| Algorithm                         | 128  | 256   | 512   | 1024  | 2048  | 4096  | 8192  | 65536  |
|-----------------------------------|------|-------|-------|-------|-------|-------|-------|--------|
| pospopcnt_u16_sse_blend_popcnt_unroll8    | **8.28** | 9.84  | 10.55 | 11    | 11.58 | 11.93 | 12.13 | 12.28  |
| pospopcnt_u16_avx512_blend_popcnt_unroll8 | 7.07 | **11.25** | **16.21** | 21    | 25.49 | 27.91 | 29.73 | 31.55  |
| pospopcnt_u16_avx512_adder_forest        | 3.05 | 2.82  | 14.53 | **23.13** | **34.37** | 44.91 | 52.78 | 61.68  |
| pospopcnt_u16_avx512_harvey_seal          | 2.07 | 2.3   | 8.21  | 15.41 | 28.17 | **49.14** | **76.11** | **138.71** |

The host architecture used is a 10 nm Cannon Lake [Core i3-8121U](https://ark.intel.com/content/www/us/en/ark/products/136863/intel-core-i3-8121u-processor-4m-cache-up-to-3-20-ghz.html) with gcc (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5).

### POPCNT

stuff

### Set algebra

same performance regardless of operator

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
 * @data: An array
 * @size: Size of data in bytes
 */
uint32_t flags[16];
int STORM_pospopcnt_u16(const uint16_t* data, uint32_t len, uint32_t* flags);
```

```C
#include "libalgebra.h"

/*
 * Compute the intersection, union, or diff cardinality between pairs of bitmaps
 * @data: An array
 * @size: Size of data in bytes
 */
// Intersect cardinality
uint64_t STORM_intersect_count(const uint64_t* data1, const uint64_t* data2, const uint32_t n_len);
// Union cardinality
uint64_t STORM_union_count(const uint64_t* data1, const uint64_t* data2, const uint32_t n_len);
// Diff cardinality
uint64_t STORM_diff_count(const uint64_t* data1, const uint64_t* data2, const uint32_t n_len);
```

## How it works

On x86 CPUs ```libalgebra.h``` uses a combination of 4 different bit
population count algorithms:

* For array sizes < 512 bytes an unrolled ```POPCNT``` algorithm
is used.
* For array sizes ≥ 512 bytes an ```AVX2``` algorithm is used.
* For array sizes ≥ 1024 bytes an ```AVX512``` algorithm is used.
* For CPUs without ```POPCNT``` instruction a portable 
integer algorithm is used.

Note that ```libalgebra.h``` works on all CPUs, it checks at run-time
whether your CPU supports POPCNT, AVX2, AVX512 before using it
and it is also thread-safe.