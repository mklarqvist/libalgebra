# libalgebra

```libalgebra.h``` is a header-only C/C++ library for:
* counting the number of set bits ("population count", `popcnt`) in an array as quickly as
possible
* compute the novel "positional population count" (`pospopcnt`) statistics
* perform set algebraic operations on bitmaps including union, intersection, and diff

using specialized CPU instructions i.e.
[POPCNT](https://en.wikipedia.org/wiki/SSE4#POPCNT_and_LZCNT),
[SSE4.2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions),
[AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions),
[AVX512BW](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions),
[NEON](https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_.28NEON.29).
```libalgebra.h``` has been tested successfully using the GCC,
Clang and MSVC compilers.

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