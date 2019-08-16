// License for libalgebra.h
/*
* Copyright (c) 2019 Marcus D. R. Klarqvist
* Author(s): Marcus D. R. Klarqvist
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
// License for pospopcnt.h
/*
* Copyright (c) 2019
* Author(s): Marcus D. R. Klarqvist, Wojciech Muła, and Daniel Lemire
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
// License for libpopcnt.h
/*
 * libpopcnt.h - C/C++ library for counting the number of 1 bits (bit
 * population count) in an array as quickly as possible using
 * specialized CPU instructions i.e. POPCNT, AVX2, AVX512, NEON.
 *
 * Copyright (c) 2016 - 2018, Kim Walisch
 * Copyright (c) 2016 - 2018, Wojciech Muła
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef LIBALGEBRA_H_9827563662203
#define LIBALGEBRA_H_9827563662203

/* *************************************
*  Includes
***************************************/
#include <stdint.h>
#include <assert.h>
#include <memory.h>
#include <string.h>
#include <math.h>

// Safety
#if !(defined(__APPLE__)) && !(defined(__FreeBSD__))
#include <malloc.h>  // this should never be needed but there are some reports that it is needed.
#endif

/* *************************************
 *  Support.
 * 
 *  These subroutines and definitions are taken from the CRoaring repo
 *  by Daniel Lemire et al. available under the Apache 2.0 License
 *  (same as libintersect.h):
 *  https://github.com/RoaringBitmap/CRoaring/ 
 ***************************************/
#if defined(__SIZEOF_LONG_LONG__) && __SIZEOF_LONG_LONG__ != 8
#error This code assumes 64-bit long longs (by use of the GCC intrinsics). Your system is not currently supported.
#endif

/* ===   Compiler specifics   === */

#if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* >= C99 */
#  define STORM_RESTRICT   restrict
#else
/* note : it might be useful to define __restrict or STORM_RESTRICT for some C++ compilers */
#  define STORM_RESTRICT   /* disable */
#endif

#include <x86intrin.h>

/****************************
*  Memory management
* 
*  The subroutines aligned_malloc and aligned_free had to be renamed to
*  STORM_aligned_malloc and STORM_aligned_free to prevent clashing with the
*  same subroutines in Roaring. These subroutines are included here
*  since there is no hard dependency on using Roaring bitmaps.
****************************/
// portable version of  posix_memalign
#ifndef STORM_aligned_malloc
static 
void* STORM_aligned_malloc(size_t alignment, size_t size) {
    void *p;
#ifdef _MSC_VER
    p = _aligned_malloc(size, alignment);
#elif defined(__MINGW32__) || defined(__MINGW64__)
    p = __mingw_aligned_malloc(size, alignment);
#else
    // somehow, if this is used before including "x86intrin.h", it creates an
    // implicit defined warning.
    if (posix_memalign(&p, alignment, size) != 0) 
        return NULL;
#endif
    return p;
}
#endif

#ifndef STORM_aligned_free
static 
void STORM_aligned_free(void* memblock) {
#ifdef _MSC_VER
    _aligned_free(memblock);
#elif defined(__MINGW32__) || defined(__MINGW64__)
    __mingw_aligned_free(memblock);
#else
    free(memblock);
#endif
}
#endif

// portable alignment
#if defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)   /* C11+ */
#  include <stdalign.h>
#  define STORM_ALIGN(n)      alignas(n)
#elif defined(__GNUC__)
#  define STORM_ALIGN(n)      __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#  define STORM_ALIGN(n)      __declspec(align(n))
#else
#  define STORM_ALIGN(n)   /* disabled */
#endif

// Taken from XXHASH
#ifdef _MSC_VER    /* Visual Studio */
#  pragma warning(disable : 4127)      /* disable: C4127: conditional expression is constant */
#  define STORM_FORCE_INLINE static __forceinline
#  define STORM_NO_INLINE static __declspec(noinline)
#else
#  if defined (__cplusplus) || defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* C99 */
#    ifdef __GNUC__
#      define STORM_FORCE_INLINE static inline __attribute__((always_inline))
#      define STORM_NO_INLINE static __attribute__((noinline))
#    else
#      define STORM_FORCE_INLINE static inline
#      define STORM_NO_INLINE static
#    endif
#  else
#    define STORM_FORCE_INLINE static
#    define STORM_NO_INLINE static
#  endif /* __STDC_VERSION__ */
#endif

// disable noise
#ifdef __GNUC__
#define WARN_UNUSED __attribute__((warn_unused_result))
#else
#define WARN_UNUSED
#endif

/*------ SIMD definitions --------*/

#define STORM_SSE_ALIGNMENT    16
#define STORM_AVX2_ALIGNMENT   32
#define STORM_AVX512_ALIGNMENT 64

/****************************
*  General checks
****************************/

#ifndef __has_builtin
  #define __has_builtin(x) 0
#endif

#ifndef __has_attribute
  #define __has_attribute(x) 0
#endif

#ifdef __GNUC__
  #define GNUC_PREREQ(x, y) \
      (__GNUC__ > x || (__GNUC__ == x && __GNUC_MINOR__ >= y))
#else
  #define GNUC_PREREQ(x, y) 0
#endif

#ifdef __clang__
  #define CLANG_PREREQ(x, y) \
      (__clang_major__ > x || (__clang_major__ == x && __clang_minor__ >= y))
#else
  #define CLANG_PREREQ(x, y) 0
#endif

#if (defined(__i386__) || \
     defined(__x86_64__) || \
     defined(_M_IX86) || \
     defined(_M_X64))
  #define X86_OR_X64
#endif

#if defined(X86_OR_X64) && \
   (defined(__cplusplus) || \
    defined(_MSC_VER) || \
   (GNUC_PREREQ(4, 2) || \
    __has_builtin(__sync_val_compare_and_swap)))
  #define HAVE_CPUID
#endif

#if GNUC_PREREQ(4, 2) || \
    __has_builtin(__builtin_popcount)
  #define HAVE_BUILTIN_POPCOUNT
#endif

#if GNUC_PREREQ(4, 2) || \
    CLANG_PREREQ(3, 0)
  #define HAVE_ASM_POPCNT
#endif

#if defined(HAVE_CPUID) && \
   (defined(HAVE_ASM_POPCNT) || \
    defined(_MSC_VER))
  #define HAVE_POPCNT
#endif

#if defined(HAVE_CPUID) && \
    GNUC_PREREQ(4, 9)
  #define HAVE_SSE42
  #define HAVE_AVX2
#endif

#if defined(HAVE_CPUID) && \
    GNUC_PREREQ(5, 0)
  #define HAVE_AVX512
#endif

#if defined(HAVE_CPUID) && \
    defined(_MSC_VER) && \
    defined(__AVX2__)
  #define HAVE_SSE42
  #define HAVE_AVX2
#endif

#if defined(HAVE_CPUID) && \
    defined(_MSC_VER) && \
    defined(__AVX512__)
  #define HAVE_AVX512
#endif

#if defined(HAVE_CPUID) && \
    CLANG_PREREQ(3, 8) && \
    __has_attribute(target) && \
   (!defined(_MSC_VER) || defined(__AVX2__)) && \
   (!defined(__apple_build_version__) || __apple_build_version__ >= 8000000)
  #define HAVE_SSE42
  #define HAVE_AVX2
  #define HAVE_AVX512
#endif

#ifdef __cplusplus
extern "C" {
#endif

/****************************
*  CPUID
****************************/
#if defined(HAVE_CPUID)

#if defined(_MSC_VER)
  #include <intrin.h>
  #include <immintrin.h>
#endif

// CPUID flags. See https://en.wikipedia.org/wiki/CPUID for more info.
/* %ecx bit flags */
#define STORM_bit_POPCNT   (1 << 23) // POPCNT instruction 
#define STORM_bit_SSE41    (1 << 19) // CPUID.01H:ECX.SSE41[Bit 19]
#define STORM_bit_SSE42    (1 << 20) // CPUID.01H:ECX.SSE41[Bit 20]

/* %ebx bit flags */
#define STORM_bit_AVX2     (1 << 5)  // CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]
#define STORM_bit_AVX512BW (1 << 30) // AVX-512 Byte and Word Instructions

/* xgetbv bit flags */
#define STORM_XSTATE_SSE (1 << 1)
#define STORM_XSTATE_YMM (1 << 2)
#define STORM_XSTATE_ZMM (7 << 5)

static  
void STORM_run_cpuid(int eax, int ecx, int* abcd) {
#if defined(_MSC_VER)
    __cpuidex(abcd, eax, ecx);
#else
    int ebx = 0;
    int edx = 0;

#if defined(__i386__) && \
    defined(__PIC__)
    /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__ ("movl %%ebx, %%edi;"
                "cpuid;"
                "xchgl %%ebx, %%edi;"
                : "=D" (ebx),
                "+a" (eax),
                "+c" (ecx),
                "=d" (edx));
#else
    __asm__ ("cpuid;"
                : "+b" (ebx),
                "+a" (eax),
                "+c" (ecx),
                "=d" (edx));
#endif

    abcd[0] = eax;
    abcd[1] = ebx;
    abcd[2] = ecx;
    abcd[3] = edx;
#endif
}

#if defined(HAVE_AVX2) || \
    defined(HAVE_AVX512)

static 
int get_xcr0() {
    int xcr0;

#if defined(_MSC_VER)
    xcr0 = (int) _xgetbv(0);
#else
    __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
#endif

    return xcr0;
}

#endif

static  
int get_cpuid() {
    int flags = 0;
    int abcd[4];

    STORM_run_cpuid(1, 0, abcd);

    // Check for POPCNT instruction
    if ((abcd[2] & STORM_bit_POPCNT) == STORM_bit_POPCNT)
        flags |= STORM_bit_POPCNT;

    // Check for SSE4.1 instruction set
    if ((abcd[2] & STORM_bit_SSE41) == STORM_bit_SSE41)
        flags |= STORM_bit_SSE41;

    // Check for SSE4.2 instruction set
    if ((abcd[2] & STORM_bit_SSE42) == STORM_bit_SSE42)
        flags |= STORM_bit_SSE42;

#if defined(HAVE_AVX2) || \
    defined(HAVE_AVX512)

    int osxsave_mask = (1 << 27);

    /* ensure OS supports extended processor state management */
    if ((abcd[2] & osxsave_mask) != osxsave_mask)
        return 0;

    int ymm_mask = STORM_XSTATE_SSE | STORM_XSTATE_YMM;
    int zmm_mask = STORM_XSTATE_SSE | STORM_XSTATE_YMM | STORM_XSTATE_ZMM;

    int xcr0 = get_xcr0();

    if ((xcr0 & ymm_mask) == ymm_mask) {
        STORM_run_cpuid(7, 0, abcd);

        if ((abcd[1] & STORM_bit_AVX2) == STORM_bit_AVX2)
            flags |= STORM_bit_AVX2;

        if ((xcr0 & zmm_mask) == zmm_mask) {
            if ((abcd[1] & STORM_bit_AVX512BW) == STORM_bit_AVX512BW)
            flags |= STORM_bit_AVX512BW;
        }
    }

#endif

  return flags;
}
#endif // defined(HAVE_CPUID)

/// Taken from libpopcnt.h
#if defined(HAVE_ASM_POPCNT) && \
    defined(__x86_64__)

STORM_FORCE_INLINE
uint64_t STORM_POPCOUNT(uint64_t x)
{
    __asm__ ("popcnt %1, %0" : "=r" (x) : "0" (x));
    return x;
}

#elif defined(HAVE_ASM_POPCNT) && \
      defined(__i386__)

STORM_FORCE_INLINE
uint32_t STORM_popcnt32(uint32_t x)
{
    __asm__ ("popcnt %1, %0" : "=r" (x) : "0" (x));
    return x;
}

STORM_FORCE_INLINE
uint64_t STORM_POPCOUNT(uint64_t x)
{
    return STORM_popcnt32((uint32_t) x) +
            STORM_popcnt32((uint32_t)(x >> 32));
}

#elif defined(_MSC_VER) && \
      defined(_M_X64)

#include <nmmintrin.h>

STORM_FORCE_INLINE
uint64_t STORM_POPCOUNT(uint64_t x) {
    return _mm_popcnt_u64(x);
}

#elif defined(_MSC_VER) && \
      defined(_M_IX86)

#include <nmmintrin.h>

STORM_FORCE_INLINE
uint64_t STORM_POPCOUNT(uint64_t x)
{
    return _mm_popcnt_u32((uint32_t) x) + 
            _mm_popcnt_u32((uint32_t)(x >> 32));
}

/* non x86 CPUs */
#elif defined(HAVE_BUILTIN_POPCOUNT)

STORM_FORCE_INLINE
uint64_t STORM_POPCOUNT(uint64_t x) {
    return __builtin_popcountll(x);
}

/* no hardware POPCNT,
 * use pure integer algorithm */
#else

// TODO: FIXME
STORM_FORCE_INLINE
uint64_t STORM_POPCOUNT(uint64_t x) {
    return STORM_popcount64(x);
}

#endif


static 
uint64_t STORM_intersect_count_unrolled(const uint64_t* STORM_RESTRICT data1, 
                                        const uint64_t* STORM_RESTRICT data2, 
                                        uint64_t size)
{
    const uint64_t limit = size - size % 4;
    uint64_t cnt = 0;
    uint64_t i   = 0;

    for (/**/; i < limit; i += 4) {
        cnt += STORM_POPCOUNT(data1[i+0] & data2[i+0]);
        cnt += STORM_POPCOUNT(data1[i+1] & data2[i+1]);
        cnt += STORM_POPCOUNT(data1[i+2] & data2[i+2]);
        cnt += STORM_POPCOUNT(data1[i+3] & data2[i+3]);
    }

    for (/**/; i < size; ++i)
        cnt += STORM_POPCOUNT(data1[i] & data2[i]);

    return cnt;
}

static 
uint64_t STORM_union_count_unrolled(const uint64_t* STORM_RESTRICT data1, 
                                    const uint64_t* STORM_RESTRICT data2, 
                                    uint64_t size)
{
    const uint64_t limit = size - size % 4;
    uint64_t cnt = 0;
    uint64_t i   = 0;

    for (/**/; i < limit; i += 4) {
        cnt += STORM_POPCOUNT(data1[i+0] | data2[i+0]);
        cnt += STORM_POPCOUNT(data1[i+1] | data2[i+1]);
        cnt += STORM_POPCOUNT(data1[i+2] | data2[i+2]);
        cnt += STORM_POPCOUNT(data1[i+3] | data2[i+3]);
    }

    for (/**/; i < size; ++i)
        cnt += STORM_POPCOUNT(data1[i] | data2[i]);

    return cnt;
}

static 
uint64_t STORM_diff_count_unrolled(const uint64_t* STORM_RESTRICT data1, 
                                   const uint64_t* STORM_RESTRICT data2, 
                                   uint64_t size)
{
    const uint64_t limit = size - size % 4;
    uint64_t cnt = 0;
    uint64_t i   = 0;

    for (/**/; i < limit; i += 4) {
        cnt += STORM_POPCOUNT(data1[i+0] ^ data2[i+0]);
        cnt += STORM_POPCOUNT(data1[i+1] ^ data2[i+1]);
        cnt += STORM_POPCOUNT(data1[i+2] ^ data2[i+2]);
        cnt += STORM_POPCOUNT(data1[i+3] ^ data2[i+3]);
    }

    for (/**/; i < size; ++i)
        cnt += STORM_POPCOUNT(data1[i] ^ data2[i]);

    return cnt;
}

static
int STORM_pospopcnt_u16_scalar_naive(const uint16_t* data, uint32_t len, uint32_t* flags) {
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((data[i] & (1 << j)) >> j);
        }
    }

    return 0;
}

#ifndef _MSC_VER

STORM_FORCE_INLINE
uint64_t STORM_pospopcnt_umul128(uint64_t a, uint64_t b, uint64_t* hi) {
    unsigned __int128 x = (unsigned __int128)a * (unsigned __int128)b;
    *hi = (uint64_t)(x >> 64);
    return (uint64_t)x;
}

STORM_FORCE_INLINE
uint64_t STORM_pospopcnt_loadu_u64(const void* ptr) {
    uint64_t data;
    memcpy(&data, ptr, sizeof(data));
    return data;
}

// By @aqrit (https://github.com/aqrit)
// @see: https://gist.github.com/aqrit/c729815b0165c139d0bac642ab7ee104
static
int STORM_pospopcnt_u16_scalar_umul128_unroll2(const uint16_t* in, uint32_t n, uint32_t* out) {
    while (n >= 8) {
        uint64_t counter_a = 0; // 4 packed 12-bit counters
        uint64_t counter_b = 0;
        uint64_t counter_c = 0;
        uint64_t counter_d = 0;

        // end before overflowing the counters
        uint32_t len = ((n < 0x0FFF) ? n : 0x0FFF) & ~7;
        n -= len;
        for (const uint16_t* end = &in[len]; in != end; in += 8) {
            const uint64_t mask_a = UINT64_C(0x1111111111111111);
            const uint64_t mask_b = mask_a + mask_a;
            const uint64_t mask_c = mask_b + mask_b;
            const uint64_t mask_0001 = UINT64_C(0x0001000100010001);
            const uint64_t mask_cnts = UINT64_C(0x000000F00F00F00F);

            uint64_t v0 = STORM_pospopcnt_loadu_u64(&in[0]);
            uint64_t v1 = STORM_pospopcnt_loadu_u64(&in[4]);

            uint64_t a = (v0 & mask_a) + (v1 & mask_a);
            uint64_t b = ((v0 & mask_b) + (v1 & mask_b)) >> 1;
            uint64_t c = ((v0 & mask_c) + (v1 & mask_c)) >> 2;
            uint64_t d = ((v0 >> 3) & mask_a) + ((v1 >> 3) & mask_a);

            uint64_t hi;
            a = STORM_pospopcnt_umul128(a, mask_0001, &hi);
            a += hi; // broadcast 4-bit counts
            b = STORM_pospopcnt_umul128(b, mask_0001, &hi);
            b += hi;
            c = STORM_pospopcnt_umul128(c, mask_0001, &hi);
            c += hi;
            d = STORM_pospopcnt_umul128(d, mask_0001, &hi);
            d += hi;

            counter_a += a & mask_cnts;
            counter_b += b & mask_cnts;
            counter_c += c & mask_cnts;
            counter_d += d & mask_cnts;
        }

        out[0] += counter_a & 0x0FFF;
        out[1] += counter_b & 0x0FFF;
        out[2] += counter_c & 0x0FFF;
        out[3] += counter_d & 0x0FFF;
        out[4] += (counter_a >> 36);
        out[5] += (counter_b >> 36);
        out[6] += (counter_c >> 36);
        out[7] += (counter_d >> 36);
        out[8] += (counter_a >> 24) & 0x0FFF;
        out[9] += (counter_b >> 24) & 0x0FFF;
        out[10] += (counter_c >> 24) & 0x0FFF;
        out[11] += (counter_d >> 24) & 0x0FFF;
        out[12] += (counter_a >> 12) & 0x0FFF;
        out[13] += (counter_b >> 12) & 0x0FFF;
        out[14] += (counter_c >> 12) & 0x0FFF;
        out[15] += (counter_d >> 12) & 0x0FFF;
    }

    // assert(n < 8)
    if (n != 0) {
        uint64_t tail_counter_a = 0;
        uint64_t tail_counter_b = 0;
        do { // zero-extend a bit to 8-bits (emulate pdep) then accumulate
            const uint64_t mask_01 = UINT64_C(0x0101010101010101);
            const uint64_t magic   = UINT64_C(0x0000040010004001); // 1+(1<<14)+(1<<28)+(1<<42)
            uint64_t x = *in++;
            tail_counter_a += ((x & 0x5555) * magic) & mask_01; // 0101010101010101
            tail_counter_b += (((x >> 1) & 0x5555) * magic) & mask_01;
        } while (--n);

        out[0]  += tail_counter_a & 0xFF;
        out[8]  += (tail_counter_a >>  8) & 0xFF;
        out[2]  += (tail_counter_a >> 16) & 0xFF;
        out[10] += (tail_counter_a >> 24) & 0xFF;
        out[4]  += (tail_counter_a >> 32) & 0xFF;
        out[12] += (tail_counter_a >> 40) & 0xFF;
        out[6]  += (tail_counter_a >> 48) & 0xFF;
        out[14] += (tail_counter_a >> 56) & 0xFF;
        out[1]  += tail_counter_b & 0xFF;
        out[9]  += (tail_counter_b >>  8) & 0xFF;
        out[3]  += (tail_counter_b >> 16) & 0xFF;
        out[11] += (tail_counter_b >> 24) & 0xFF;
        out[5]  += (tail_counter_b >> 32) & 0xFF;
        out[13] += (tail_counter_b >> 40) & 0xFF;
        out[7]  += (tail_counter_b >> 48) & 0xFF;
        out[15] += (tail_counter_b >> 56) & 0xFF;
    }

    return 0;
}
#endif

/*
 * This uses fewer arithmetic operations than any other known
 * implementation on machines with fast multiplication.
 * It uses 12 arithmetic operations, one of which is a multiply.
 * http://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation
 */
STORM_FORCE_INLINE
uint64_t STORM_popcount64(uint64_t x)
{
    uint64_t m1 = 0x5555555555555555ll;
    uint64_t m2 = 0x3333333333333333ll;
    uint64_t m4 = 0x0F0F0F0F0F0F0F0Fll;
    uint64_t h01 = 0x0101010101010101ll;

    x -= (x >> 1) & m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;

    return (x * h01) >> 56;
}


static
const uint8_t STORM_popcnt_lookup8bit[256] = {
	/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
	/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
	/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
	/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
	/* 10 */ 1, /* 11 */ 2, /* 12 */ 2, /* 13 */ 3,
	/* 14 */ 2, /* 15 */ 3, /* 16 */ 3, /* 17 */ 4,
	/* 18 */ 2, /* 19 */ 3, /* 1a */ 3, /* 1b */ 4,
	/* 1c */ 3, /* 1d */ 4, /* 1e */ 4, /* 1f */ 5,
	/* 20 */ 1, /* 21 */ 2, /* 22 */ 2, /* 23 */ 3,
	/* 24 */ 2, /* 25 */ 3, /* 26 */ 3, /* 27 */ 4,
	/* 28 */ 2, /* 29 */ 3, /* 2a */ 3, /* 2b */ 4,
	/* 2c */ 3, /* 2d */ 4, /* 2e */ 4, /* 2f */ 5,
	/* 30 */ 2, /* 31 */ 3, /* 32 */ 3, /* 33 */ 4,
	/* 34 */ 3, /* 35 */ 4, /* 36 */ 4, /* 37 */ 5,
	/* 38 */ 3, /* 39 */ 4, /* 3a */ 4, /* 3b */ 5,
	/* 3c */ 4, /* 3d */ 5, /* 3e */ 5, /* 3f */ 6,
	/* 40 */ 1, /* 41 */ 2, /* 42 */ 2, /* 43 */ 3,
	/* 44 */ 2, /* 45 */ 3, /* 46 */ 3, /* 47 */ 4,
	/* 48 */ 2, /* 49 */ 3, /* 4a */ 3, /* 4b */ 4,
	/* 4c */ 3, /* 4d */ 4, /* 4e */ 4, /* 4f */ 5,
	/* 50 */ 2, /* 51 */ 3, /* 52 */ 3, /* 53 */ 4,
	/* 54 */ 3, /* 55 */ 4, /* 56 */ 4, /* 57 */ 5,
	/* 58 */ 3, /* 59 */ 4, /* 5a */ 4, /* 5b */ 5,
	/* 5c */ 4, /* 5d */ 5, /* 5e */ 5, /* 5f */ 6,
	/* 60 */ 2, /* 61 */ 3, /* 62 */ 3, /* 63 */ 4,
	/* 64 */ 3, /* 65 */ 4, /* 66 */ 4, /* 67 */ 5,
	/* 68 */ 3, /* 69 */ 4, /* 6a */ 4, /* 6b */ 5,
	/* 6c */ 4, /* 6d */ 5, /* 6e */ 5, /* 6f */ 6,
	/* 70 */ 3, /* 71 */ 4, /* 72 */ 4, /* 73 */ 5,
	/* 74 */ 4, /* 75 */ 5, /* 76 */ 5, /* 77 */ 6,
	/* 78 */ 4, /* 79 */ 5, /* 7a */ 5, /* 7b */ 6,
	/* 7c */ 5, /* 7d */ 6, /* 7e */ 6, /* 7f */ 7,
	/* 80 */ 1, /* 81 */ 2, /* 82 */ 2, /* 83 */ 3,
	/* 84 */ 2, /* 85 */ 3, /* 86 */ 3, /* 87 */ 4,
	/* 88 */ 2, /* 89 */ 3, /* 8a */ 3, /* 8b */ 4,
	/* 8c */ 3, /* 8d */ 4, /* 8e */ 4, /* 8f */ 5,
	/* 90 */ 2, /* 91 */ 3, /* 92 */ 3, /* 93 */ 4,
	/* 94 */ 3, /* 95 */ 4, /* 96 */ 4, /* 97 */ 5,
	/* 98 */ 3, /* 99 */ 4, /* 9a */ 4, /* 9b */ 5,
	/* 9c */ 4, /* 9d */ 5, /* 9e */ 5, /* 9f */ 6,
	/* a0 */ 2, /* a1 */ 3, /* a2 */ 3, /* a3 */ 4,
	/* a4 */ 3, /* a5 */ 4, /* a6 */ 4, /* a7 */ 5,
	/* a8 */ 3, /* a9 */ 4, /* aa */ 4, /* ab */ 5,
	/* ac */ 4, /* ad */ 5, /* ae */ 5, /* af */ 6,
	/* b0 */ 3, /* b1 */ 4, /* b2 */ 4, /* b3 */ 5,
	/* b4 */ 4, /* b5 */ 5, /* b6 */ 5, /* b7 */ 6,
	/* b8 */ 4, /* b9 */ 5, /* ba */ 5, /* bb */ 6,
	/* bc */ 5, /* bd */ 6, /* be */ 6, /* bf */ 7,
	/* c0 */ 2, /* c1 */ 3, /* c2 */ 3, /* c3 */ 4,
	/* c4 */ 3, /* c5 */ 4, /* c6 */ 4, /* c7 */ 5,
	/* c8 */ 3, /* c9 */ 4, /* ca */ 4, /* cb */ 5,
	/* cc */ 4, /* cd */ 5, /* ce */ 5, /* cf */ 6,
	/* d0 */ 3, /* d1 */ 4, /* d2 */ 4, /* d3 */ 5,
	/* d4 */ 4, /* d5 */ 5, /* d6 */ 5, /* d7 */ 6,
	/* d8 */ 4, /* d9 */ 5, /* da */ 5, /* db */ 6,
	/* dc */ 5, /* dd */ 6, /* de */ 6, /* df */ 7,
	/* e0 */ 3, /* e1 */ 4, /* e2 */ 4, /* e3 */ 5,
	/* e4 */ 4, /* e5 */ 5, /* e6 */ 5, /* e7 */ 6,
	/* e8 */ 4, /* e9 */ 5, /* ea */ 5, /* eb */ 6,
	/* ec */ 5, /* ed */ 6, /* ee */ 6, /* ef */ 7,
	/* f0 */ 4, /* f1 */ 5, /* f2 */ 5, /* f3 */ 6,
	/* f4 */ 5, /* f5 */ 6, /* f6 */ 6, /* f7 */ 7,
	/* f8 */ 5, /* f9 */ 6, /* fa */ 6, /* fb */ 7,
	/* fc */ 6, /* fd */ 7, /* fe */ 7, /* ff */ 8
};

/****************************
*  SSE4.1 functions
****************************/

#if defined(HAVE_SSE42)

#include <immintrin.h>

#ifndef STORM_POPCOUNT_SSE4
#define STORM_POPCOUNT_SSE4(A, B) {               \
    A += STORM_POPCOUNT(_mm_extract_epi64(B, 0)); \
    A += STORM_POPCOUNT(_mm_extract_epi64(B, 1)); \
}
#endif

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
STORM_FORCE_INLINE  
uint64_t STORM_POPCOUNT_SSE(const __m128i n) {
    return(STORM_POPCOUNT(_mm_cvtsi128_si64(n)) + 
           STORM_POPCOUNT(_mm_cvtsi128_si64(_mm_unpackhi_epi64(n, n))));
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
STORM_FORCE_INLINE 
void STORM_CSA128(__m128i* h, __m128i* l, __m128i a, __m128i b, __m128i c) {
    __m128i u = _mm_xor_si128(a, b);
    *h = _mm_or_si128(_mm_and_si128(a, b), _mm_and_si128(u, c));
    *l = _mm_xor_si128(u, c);
}

/**
 * Carry-save adder update step.
 * @see https://en.wikipedia.org/wiki/Carry-save_adder#Technical_details
 * 
 * Steps:
 * 1)  U = *L ⊕ B
 * 2) *H = (*L ^ B) | (U ^ C)
 * 3) *L = *L ⊕ B ⊕ C = U ⊕ C
 * 
 * B and C are 16-bit staggered registers such that &C - &B = 1.
 * 
 * Example usage:
 * pospopcnt_csa_sse(&twosA, &v1, _mm_loadu_si128(data + i + 0), _mm_loadu_si128(data + i + 1));
 * 
 * @param h 
 * @param l 
 * @param b 
 * @param c  
 */
#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
STORM_FORCE_INLINE
void STORM_pospopcnt_csa_sse(__m128i* STORM_RESTRICT h, 
                             __m128i* STORM_RESTRICT l, 
                             const __m128i b, 
                             const __m128i c) 
{
    const __m128i u = _mm_xor_si128(*l, b);
    *h = _mm_or_si128(*l & b, u & c); // shift carry (sc_i).
    *l = _mm_xor_si128(u, c); // partial sum (ps).
}

// By @aqrit (https://github.com/aqrit)
// @see: https://gist.github.com/aqrit/cb52b2ac5b7d0dfe9319c09d27237bf3
#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static
int STORM_pospopcnt_u16_sse_sad(const uint16_t* data, uint32_t len, uint32_t* flag_counts) {
    const __m128i zero = _mm_setzero_si128();
    const __m128i mask_lo_byte = _mm_srli_epi16(_mm_cmpeq_epi8(zero, zero), 8);
    const __m128i mask_lo_cnt  = _mm_srli_epi16(mask_lo_byte, 2);
    const __m128i mask_bits_a  = _mm_set1_epi8(0x41); // 01000001
    const __m128i mask_bits_b  = _mm_add_epi8(mask_bits_a, mask_bits_a);
    uint32_t buffer[16];

    __m128i counterA = zero;
    __m128i counterB = zero;
    __m128i counterC = zero;
    __m128i counterD = zero;

    for (const uint16_t* end = &data[(len & ~31)]; data != end; data += 32) {
        __m128i r0 = _mm_loadu_si128((__m128i*)&data[0]);
        __m128i r1 = _mm_loadu_si128((__m128i*)&data[8]);
        __m128i r2 = _mm_loadu_si128((__m128i*)&data[16]);
        __m128i r3 = _mm_loadu_si128((__m128i*)&data[24]);
        __m128i r4, r5, r6, r7;

        // seperate LOBYTE and HIBYTE of each WORD
        // (emulate PSHUFB F,D,B,9,7,5,3,1, E,C,A,8,6,4,2,0)
        r4 = _mm_and_si128(mask_lo_byte, r0);
        r5 = _mm_and_si128(mask_lo_byte, r1);
        r6 = _mm_and_si128(mask_lo_byte, r2);
        r7 = _mm_and_si128(mask_lo_byte, r3);
        r0 = _mm_srli_epi16(r0, 8);
        r1 = _mm_srli_epi16(r1, 8);
        r2 = _mm_srli_epi16(r2, 8);
        r3 = _mm_srli_epi16(r3, 8);
        r0 = _mm_packus_epi16(r0, r4);
        r1 = _mm_packus_epi16(r1, r5);
        r2 = _mm_packus_epi16(r2, r6);
        r3 = _mm_packus_epi16(r3, r7);

        // isolate bits to count
        r4 = _mm_and_si128(mask_bits_a, r0);
        r5 = _mm_and_si128(mask_bits_a, r1);
        r6 = _mm_and_si128(mask_bits_a, r2);
        r7 = _mm_and_si128(mask_bits_a, r3);

        // horizontal sum of qwords
        r4 = _mm_sad_epu8(r4, zero);
        r5 = _mm_sad_epu8(r5, zero);
        r6 = _mm_sad_epu8(r6, zero);
        r7 = _mm_sad_epu8(r7, zero);

        // sum 6-bit counts
        r4 = _mm_add_epi16(r4,r5);
        r4 = _mm_add_epi16(r4,r6);
        r4 = _mm_add_epi16(r4,r7);

        // unpack 6-bit counts to 32-bits
        r5 = _mm_and_si128(mask_lo_cnt, r4);
        r4 = _mm_srli_epi16(r4, 6);
        r4 = _mm_packs_epi32(r4, r5);

        // accumulate
        counterA = _mm_add_epi32(counterA, r4);

        // do it again...
        r4 = _mm_and_si128(mask_bits_b, r0);
        r5 = _mm_and_si128(mask_bits_b, r1);
        r6 = _mm_and_si128(mask_bits_b, r2);
        r7 = _mm_and_si128(mask_bits_b, r3);

        r4 = _mm_sad_epu8(r4, zero);
        r5 = _mm_sad_epu8(r5, zero);
        r6 = _mm_sad_epu8(r6, zero);
        r7 = _mm_sad_epu8(r7, zero);

        r4 = _mm_add_epi16(r4,r5);
        r4 = _mm_add_epi16(r4,r6);
        r4 = _mm_add_epi16(r4,r7);

        r5 = _mm_avg_epu8(zero, r4); // shift right 1
        r5 = _mm_and_si128(r5, mask_lo_cnt);
        r4 = _mm_srli_epi16(r4, 7);
        r4 = _mm_packs_epi32(r4, r5);

        counterB = _mm_add_epi32(counterB, r4); // accumulate

        // rotate right 4
        r4 = _mm_slli_epi16(r0, 12);
        r5 = _mm_slli_epi16(r1, 12);
        r6 = _mm_slli_epi16(r2, 12);
        r7 = _mm_slli_epi16(r3, 12);
        r0 = _mm_srli_epi16(r0, 4);
        r1 = _mm_srli_epi16(r1, 4);
        r2 = _mm_srli_epi16(r2, 4);
        r3 = _mm_srli_epi16(r3, 4);
        r0 = _mm_or_si128(r0, r4);
        r1 = _mm_or_si128(r1, r5);
        r2 = _mm_or_si128(r2, r6);
        r3 = _mm_or_si128(r3, r7);

        // do it again...
        r4 = _mm_and_si128(mask_bits_a, r0);
        r5 = _mm_and_si128(mask_bits_a, r1);
        r6 = _mm_and_si128(mask_bits_a, r2);
        r7 = _mm_and_si128(mask_bits_a, r3);

        r4 = _mm_sad_epu8(r4, zero);
        r5 = _mm_sad_epu8(r5, zero);
        r6 = _mm_sad_epu8(r6, zero);
        r7 = _mm_sad_epu8(r7, zero);

        r4 = _mm_add_epi16(r4,r5);
        r4 = _mm_add_epi16(r4,r6);
        r4 = _mm_add_epi16(r4,r7);

        r5 = _mm_and_si128(mask_lo_cnt, r4);
        r4 = _mm_srli_epi16(r4, 6);
        r4 = _mm_packs_epi32(r4, r5);

        counterC = _mm_add_epi32(counterC, r4); // accumulate

        // do it again...
        r0 = _mm_and_si128(r0, mask_bits_b);
        r1 = _mm_and_si128(r1, mask_bits_b);
        r2 = _mm_and_si128(r2, mask_bits_b);
        r3 = _mm_and_si128(r3, mask_bits_b);

        r0 = _mm_sad_epu8(r0, zero);
        r1 = _mm_sad_epu8(r1, zero);
        r2 = _mm_sad_epu8(r2, zero);
        r3 = _mm_sad_epu8(r3, zero);

        r0 = _mm_add_epi16(r0,r1);
        r0 = _mm_add_epi16(r0,r2);
        r0 = _mm_add_epi16(r0,r3);

        r1 = _mm_avg_epu8(zero, r0);
        r1 = _mm_and_si128(r1, mask_lo_cnt);
        r0 = _mm_srli_epi16(r0, 7);
        r0 = _mm_packs_epi32(r0, r1);

        counterD = _mm_add_epi32(counterD, r0); // accumulate
    }

    // transpose then store counters
    __m128i counter_1098 = _mm_unpackhi_epi32(counterA, counterB);
    __m128i counter_76FE = _mm_unpacklo_epi32(counterA, counterB);
    __m128i counter_32BA = _mm_unpacklo_epi32(counterC, counterD);
    __m128i counter_54DC = _mm_unpackhi_epi32(counterC, counterD);
    __m128i counter_7654 = _mm_unpackhi_epi64(counter_54DC, counter_76FE);
    __m128i counter_FEDC = _mm_unpacklo_epi64(counter_54DC, counter_76FE);
    __m128i counter_3210 = _mm_unpackhi_epi64(counter_1098, counter_32BA);
    __m128i counter_BA98 = _mm_unpacklo_epi64(counter_1098, counter_32BA);

    
    _mm_storeu_si128((__m128i*)&buffer[0], counter_3210);
    _mm_storeu_si128((__m128i*)&buffer[4], counter_7654);
    _mm_storeu_si128((__m128i*)&buffer[8], counter_BA98);
    _mm_storeu_si128((__m128i*)&buffer[12], counter_FEDC);
    for (int i = 0; i < 16; ++i) flag_counts[i] += buffer[i];

    // scalar tail loop
    int tail = len & 31;
    if (tail != 0) {
        uint64_t countsA = 0;
        uint64_t countsB = 0;
        do {
            // zero-extend a bit to 8-bits then accumulate
            // (emulate pdep)
            const uint64_t mask_01 = UINT64_C(0x0101010101010101);// 100000001000000010000000100000001000000010000000100000001
            const uint64_t magic   = UINT64_C(0x0000040010004001);// 000000000000001000000000000010000000000000100000000000001
                                                                  // 1+(1<<14)+(1<<28)+(1<<42)
            uint64_t x = *data++;
            countsA += ((x & 0x5555) * magic) & mask_01; // 0101010101010101
            countsB += (((x >> 1) & 0x5555) * magic) & mask_01;
        } while (--tail);

        // transpose then store counters
        flag_counts[0]  += countsA & 0xFF;
        flag_counts[8]  += (countsA >>  8) & 0xFF;
        flag_counts[2]  += (countsA >> 16) & 0xFF;
        flag_counts[10] += (countsA >> 24) & 0xFF;
        flag_counts[4]  += (countsA >> 32) & 0xFF;
        flag_counts[12] += (countsA >> 40) & 0xFF;
        flag_counts[6]  += (countsA >> 48) & 0xFF;
        flag_counts[14] += (countsA >> 56) & 0xFF;
        flag_counts[1]  += countsB & 0xFF;
        flag_counts[9]  += (countsB >>  8) & 0xFF;
        flag_counts[3]  += (countsB >> 16) & 0xFF;
        flag_counts[11] += (countsB >> 24) & 0xFF;
        flag_counts[5]  += (countsB >> 32) & 0xFF;
        flag_counts[13] += (countsB >> 40) & 0xFF;
        flag_counts[7]  += (countsB >> 48) & 0xFF;
        flag_counts[15] += (countsB >> 56) & 0xFF;
    }

    return 0;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static
int STORM_pospopcnt_u16_sse_blend_popcnt_unroll8(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m128i* data_vectors = (const __m128i*)(array);
    const uint32_t n_cycles = len / 8;

    size_t i = 0;
    for (/**/; i + 8 <= n_cycles; i += 8) {
#define L(p) __m128i v##p = _mm_loadu_si128(data_vectors+i+p);
        L(0) L(1) L(2) L(3)
        L(4) L(5) L(6) L(7)

#define U0(p,k) __m128i input##p = _mm_or_si128(_mm_and_si128(v##p, _mm_set1_epi16(0x00FF)), _mm_slli_epi16(v##k, 8));
#define U1(p,k) __m128i input##k = _mm_or_si128(_mm_and_si128(v##p, _mm_set1_epi16(0xFF00)), _mm_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)

        U(0,1) U(2,3) U(4,5) U(6,7)
        
        for (int i = 0; i < 8; ++i) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A(2, 3) A(4,5) A(6, 7)

#define P0(p) input##p = _mm_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)

            P(0,1) P(2, 3) P(4,5) P(6, 7)
        }
    }

    for (/**/; i + 4 <= n_cycles; i += 4) {
        L(0) L(1) L(2) L(3)
        U(0,1) U(2,3)
        
        for (int i = 0; i < 8; ++i) {
            A(0,1) A(2, 3)
            P(0,1) P(2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; ++i) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 8;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P
    return 0;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static
int STORM_pospopcnt_u16_sse_harvey_seal(const uint16_t* array, uint32_t len, uint32_t* flags) {
    for (uint32_t i = len - (len % (16 * 8)); i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

    const __m128i* data = (const __m128i*)array;
    size_t size = len / 8;
    __m128i v1  = _mm_setzero_si128();
    __m128i v2  = _mm_setzero_si128();
    __m128i v4  = _mm_setzero_si128();
    __m128i v8  = _mm_setzero_si128();
    __m128i v16 = _mm_setzero_si128();
    __m128i twosA, twosB, foursA, foursB, eightsA, eightsB;

    const uint64_t limit = size - size % 16;
    uint64_t i = 0;
    uint16_t buffer[8];
    __m128i counter[16];

    while (i < limit) {        
        for (size_t i = 0; i < 16; ++i) {
            counter[i] = _mm_setzero_si128();
        }

        size_t thislimit = limit;
        if (thislimit - i >= (1 << 16))
            thislimit = i + (1 << 16) - 1;

        for (/**/; i < thislimit; i += 16) {
#define U(pos) {                     \
    counter[pos] = _mm_add_epi16(counter[pos], _mm_and_si128(v16, _mm_set1_epi16(1))); \
    v16 = _mm_srli_epi16(v16, 1); \
}
            STORM_pospopcnt_csa_sse(&twosA,  &v1, _mm_loadu_si128(data + i +  0), _mm_loadu_si128(data + i +  1));
            STORM_pospopcnt_csa_sse(&twosB,  &v1, _mm_loadu_si128(data + i +  2), _mm_loadu_si128(data + i +  3));
            STORM_pospopcnt_csa_sse(&foursA, &v2, twosA, twosB);
            STORM_pospopcnt_csa_sse(&twosA,  &v1, _mm_loadu_si128(data + i +  4), _mm_loadu_si128(data + i +  5));
            STORM_pospopcnt_csa_sse(&twosB,  &v1, _mm_loadu_si128(data + i +  6), _mm_loadu_si128(data + i +  7));
            STORM_pospopcnt_csa_sse(&foursB, &v2, twosA, twosB);
            STORM_pospopcnt_csa_sse(&eightsA,&v4, foursA, foursB);
            STORM_pospopcnt_csa_sse(&twosA,  &v1, _mm_loadu_si128(data + i +  8),  _mm_loadu_si128(data + i +  9));
            STORM_pospopcnt_csa_sse(&twosB,  &v1, _mm_loadu_si128(data + i + 10),  _mm_loadu_si128(data + i + 11));
            STORM_pospopcnt_csa_sse(&foursA, &v2, twosA, twosB);
            STORM_pospopcnt_csa_sse(&twosA,  &v1, _mm_loadu_si128(data + i + 12),  _mm_loadu_si128(data + i + 13));
            STORM_pospopcnt_csa_sse(&twosB,  &v1, _mm_loadu_si128(data + i + 14),  _mm_loadu_si128(data + i + 15));
            STORM_pospopcnt_csa_sse(&foursB, &v2, twosA, twosB);
            STORM_pospopcnt_csa_sse(&eightsB,&v4, foursA, foursB);
            U(0) U(1) U(2) U(3) U(4) U(5) U(6) U(7) U(8) U(9) U(10) U(11) U(12) U(13) U(14) U(15) // Updates
            STORM_pospopcnt_csa_sse(&v16,    &v8, eightsA, eightsB);
#undef U
        }

        // update the counters after the last iteration
        for (size_t i = 0; i < 16; ++i) {
            counter[i] = _mm_add_epi16(counter[i], _mm_and_si128(v16, _mm_set1_epi16(1)));
            v16 = _mm_srli_epi16(v16, 1);
        }
        
        for (size_t i = 0; i < 16; ++i) {
            _mm_storeu_si128((__m128i*)buffer, counter[i]);
            for (size_t z = 0; z < 8; z++) {
                flags[i] += 16 * (uint32_t)buffer[z];
            }
        }
    }

    _mm_storeu_si128((__m128i*)buffer, v1);
    for (size_t i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((buffer[i] & (1 << j)) >> j);
        }
    }

    _mm_storeu_si128((__m128i*)buffer, v2);
    for (size_t i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += 2 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    _mm_storeu_si128((__m128i*)buffer, v4);
    for (size_t i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += 4 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    _mm_storeu_si128((__m128i*)buffer, v8);
    for (size_t i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += 8 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    return 0;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static 
uint64_t STORM_intersect_count_csa_sse4(const __m128i* STORM_RESTRICT data1, 
                                        const __m128i* STORM_RESTRICT data2, 
                                        uint64_t size)
{
    __m128i ones     = _mm_setzero_si128();
    __m128i twos     = _mm_setzero_si128();
    __m128i fours    = _mm_setzero_si128();
    __m128i eights   = _mm_setzero_si128();
    __m128i sixteens = _mm_setzero_si128();
    __m128i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t cnt64 = 0;

#define LOAD(a) (_mm_loadu_si128(&data1[i+a]) & _mm_loadu_si128(&data2[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA128(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA128(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA128(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA128(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA128(&sixteens,&eights, eights,eightsA,eightsB);

        cnt64 += STORM_POPCOUNT_SSE(sixteens);
    }
#undef LOAD

    cnt64 <<= 4;
    cnt64 += STORM_POPCOUNT_SSE(eights) << 3;
    cnt64 += STORM_POPCOUNT_SSE(fours)  << 2;
    cnt64 += STORM_POPCOUNT_SSE(twos)   << 1;
    cnt64 += STORM_POPCOUNT_SSE(ones)   << 0;

    for (/**/; i < size; ++i)
        cnt64 = STORM_POPCOUNT_SSE(_mm_loadu_si128(&data1[i]) & _mm_loadu_si128(&data2[i]));

    return cnt64;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static 
uint64_t STORM_union_count_csa_sse4(const __m128i* STORM_RESTRICT data1, 
                                    const __m128i* STORM_RESTRICT data2, 
                                    uint64_t size)
{
    __m128i ones     = _mm_setzero_si128();
    __m128i twos     = _mm_setzero_si128();
    __m128i fours    = _mm_setzero_si128();
    __m128i eights   = _mm_setzero_si128();
    __m128i sixteens = _mm_setzero_si128();
    __m128i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t cnt64 = 0;

#define LOAD(a) (_mm_loadu_si128(&data1[i+a]) | _mm_loadu_si128(&data2[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA128(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA128(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA128(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA128(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA128(&sixteens,&eights, eights,eightsA,eightsB);

        cnt64 += STORM_POPCOUNT_SSE(sixteens);
    }
#undef LOAD

    cnt64 <<= 4;
    cnt64 += STORM_POPCOUNT_SSE(eights) << 3;
    cnt64 += STORM_POPCOUNT_SSE(fours)  << 2;
    cnt64 += STORM_POPCOUNT_SSE(twos)   << 1;
    cnt64 += STORM_POPCOUNT_SSE(ones)   << 0;

    for (/**/; i < size; ++i)
        cnt64 = STORM_POPCOUNT_SSE(_mm_loadu_si128(&data1[i]) | _mm_loadu_si128(&data2[i]));

    return cnt64;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static 
uint64_t STORM_diff_count_csa_sse4(const __m128i* STORM_RESTRICT data1, 
                                   const __m128i* STORM_RESTRICT data2, 
                                   uint64_t size)
{
    __m128i ones     = _mm_setzero_si128();
    __m128i twos     = _mm_setzero_si128();
    __m128i fours    = _mm_setzero_si128();
    __m128i eights   = _mm_setzero_si128();
    __m128i sixteens = _mm_setzero_si128();
    __m128i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t cnt64 = 0;

#define LOAD(a) (_mm_loadu_si128(&data1[i+a]) ^ _mm_loadu_si128(&data2[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA128(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA128(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA128(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA128(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA128(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA128(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA128(&sixteens,&eights, eights,eightsA,eightsB);

        cnt64 += STORM_POPCOUNT_SSE(sixteens);
    }
#undef LOAD

    cnt64 <<= 4;
    cnt64 += STORM_POPCOUNT_SSE(eights) << 3;
    cnt64 += STORM_POPCOUNT_SSE(fours)  << 2;
    cnt64 += STORM_POPCOUNT_SSE(twos)   << 1;
    cnt64 += STORM_POPCOUNT_SSE(ones)   << 0;

    for (/**/; i < size; ++i)
        cnt64 = STORM_POPCOUNT_SSE(_mm_loadu_si128(&data1[i]) ^ _mm_loadu_si128(&data2[i]));

    return cnt64;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static 
uint64_t STORM_intersect_count_sse4(const uint64_t* STORM_RESTRICT b1, 
                            const uint64_t* STORM_RESTRICT b2, 
                            const uint32_t n_ints) 
{
    uint64_t count = 0;
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    count += STORM_intersect_count_csa_sse4(r1, r2, n_cycles);

    for (int i = n_cycles*2; i < n_ints; ++i) {
        count += STORM_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static 
uint64_t STORM_union_count_sse4(const uint64_t* STORM_RESTRICT b1, 
                                const uint64_t* STORM_RESTRICT b2, 
                                const uint32_t n_ints) 
{
    uint64_t count = 0;
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    count += STORM_union_count_csa_sse4(r1, r2, n_cycles);

    for (int i = n_cycles*2; i < n_ints; ++i) {
        count += STORM_POPCOUNT(b1[i] | b2[i]);
    }

    return(count);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static 
uint64_t STORM_diff_count_sse4(const uint64_t* STORM_RESTRICT b1, 
                            const uint64_t* STORM_RESTRICT b2, 
                            const uint32_t n_ints) 
{
    uint64_t count = 0;
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    count += STORM_diff_count_csa_sse4(r1, r2, n_cycles);

    for (int i = n_cycles*2; i < n_ints; ++i) {
        count += STORM_POPCOUNT(b1[i] ^ b2[i]);
    }

    return(count);
}
#endif

/****************************
*  AVX256 functions
****************************/

#if defined(HAVE_AVX2)

#include <immintrin.h>

#ifndef STORM_POPCOUNT_AVX2
#define STORM_POPCOUNT_AVX2(A, B) {                  \
    A += STORM_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += STORM_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += STORM_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += STORM_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
STORM_FORCE_INLINE 
void STORM_CSA256(__m256i* h, __m256i* l, __m256i a, __m256i b, __m256i c) {
    __m256i u = _mm256_xor_si256(a, b);
    *h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    *l = _mm256_xor_si256(u, c);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
STORM_FORCE_INLINE
void STORM_pospopcnt_csa_avx2(__m256i* STORM_RESTRICT h, 
                              __m256i* STORM_RESTRICT l, 
                              const __m256i b, 
                              const __m256i c) 
{
    const __m256i u = _mm256_xor_si256(*l, b);
    *h = _mm256_or_si256(*l & b, u & c);
    *l = _mm256_xor_si256(u, c);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static
int STORM_pospopcnt_u16_avx2_blend_popcnt_unroll8(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m256i* data_vectors = (const __m256i*)(array);
    const uint32_t n_cycles = len / 16;

    size_t i = 0;
    for (/**/; i + 8 <= n_cycles; i += 8) {
#define L(p) __m256i v##p = _mm256_loadu_si256(data_vectors+i+p);
        L(0) L(1) L(2) L(3)
        L(4) L(5) L(6) L(7) 
        
#define U0(p,k) __m256i input##p = _mm256_or_si256(_mm256_and_si256(v##p, _mm256_set1_epi16(0x00FF)), _mm256_slli_epi16(v##k, 8));
#define U1(p,k) __m256i input##k = _mm256_or_si256(_mm256_and_si256(v##p, _mm256_set1_epi16(0xFF00)), _mm256_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)
       U(0,1) U(2, 3) U(4, 5) U(6, 7)
        
        for (int i = 0; i < 8; ++i) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A(2, 3) A(4, 5) A(6, 7)

#define P0(p) input##p = _mm256_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)
            P(0,1) P(2, 3) P(4, 5) P(6, 7)
        }
    }

    for (/**/; i + 4 <= n_cycles; i += 4) {
        L(0) L(1) L(2) L(3)
        U(0,1) U(2, 3)
        
        for (int i = 0; i < 8; ++i) {
            A(0,1) A( 2, 3)
            P(0,1) P( 2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; ++i) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 16;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P

    return 0;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
int STORM_pospopcnt_u16_avx2_harvey_seal(const uint16_t* array, uint32_t len, uint32_t* flags) {
    for (uint32_t i = len - (len % (16 * 16)); i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

    const __m256i* data = (const __m256i*)array;
    size_t size = len / 16;
    __m256i v1  = _mm256_setzero_si256();
    __m256i v2  = _mm256_setzero_si256();
    __m256i v4  = _mm256_setzero_si256();
    __m256i v8  = _mm256_setzero_si256();
    __m256i v16 = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    const uint64_t limit = size - size % 16;
    uint64_t i = 0;
    uint16_t buffer[16];
    __m256i counter[16];
    const __m256i one = _mm256_set1_epi16(1);

    while (i < limit) {        
        for (size_t i = 0; i < 16; ++i) {
            counter[i] = _mm256_setzero_si256();
        }

        size_t thislimit = limit;
        if (thislimit - i >= (1 << 16))
            thislimit = i + (1 << 16) - 1;

        for (/**/; i < thislimit; i += 16) {
#define U(pos) {                     \
    counter[pos] = _mm256_add_epi16(counter[pos], _mm256_and_si256(v16, one)); \
    v16 = _mm256_srli_epi16(v16, 1); \
}
            STORM_pospopcnt_csa_avx2(&twosA,  &v1, _mm256_loadu_si256(data + i +  0), _mm256_loadu_si256(data + i +  1));
            STORM_pospopcnt_csa_avx2(&twosB,  &v1, _mm256_loadu_si256(data + i +  2), _mm256_loadu_si256(data + i +  3));
            STORM_pospopcnt_csa_avx2(&foursA, &v2, twosA, twosB);
            STORM_pospopcnt_csa_avx2(&twosA,  &v1, _mm256_loadu_si256(data + i +  4), _mm256_loadu_si256(data + i +  5));
            STORM_pospopcnt_csa_avx2(&twosB,  &v1, _mm256_loadu_si256(data + i +  6), _mm256_loadu_si256(data + i +  7));
            STORM_pospopcnt_csa_avx2(&foursB, &v2, twosA, twosB);
            STORM_pospopcnt_csa_avx2(&eightsA,&v4, foursA, foursB);
            STORM_pospopcnt_csa_avx2(&twosA,  &v1, _mm256_loadu_si256(data + i +  8),  _mm256_loadu_si256(data + i +  9));
            STORM_pospopcnt_csa_avx2(&twosB,  &v1, _mm256_loadu_si256(data + i + 10),  _mm256_loadu_si256(data + i + 11));
            STORM_pospopcnt_csa_avx2(&foursA, &v2, twosA, twosB);
            STORM_pospopcnt_csa_avx2(&twosA,  &v1, _mm256_loadu_si256(data + i + 12),  _mm256_loadu_si256(data + i + 13));
            STORM_pospopcnt_csa_avx2(&twosB,  &v1, _mm256_loadu_si256(data + i + 14),  _mm256_loadu_si256(data + i + 15));
            STORM_pospopcnt_csa_avx2(&foursB, &v2, twosA, twosB);
            STORM_pospopcnt_csa_avx2(&eightsB,&v4, foursA, foursB);
            U(0) U(1) U(2) U(3) U(4) U(5) U(6) U(7) U(8) U(9) U(10) U(11) U(12) U(13) U(14) U(15) // Updates
            STORM_pospopcnt_csa_avx2(&v16,    &v8, eightsA, eightsB);
#undef U
        }

        // update the counters after the last iteration
        for (size_t i = 0; i < 16; ++i) {
            counter[i] = _mm256_add_epi16(counter[i], _mm256_and_si256(v16, one));
            v16 = _mm256_srli_epi16(v16, 1);
        }
        
        for (size_t i = 0; i < 16; ++i) {
            _mm256_storeu_si256((__m256i*)buffer, counter[i]);
            for (size_t z = 0; z < 16; z++) {
                flags[i] += 16 * (uint32_t)buffer[z];
            }
        }
    }

    _mm256_storeu_si256((__m256i*)buffer, v1);
    for (size_t i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((buffer[i] & (1 << j)) >> j);
        }
    }

    _mm256_storeu_si256((__m256i*)buffer, v2);
    for (size_t i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += 2 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    _mm256_storeu_si256((__m256i*)buffer, v4);
    for (size_t i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += 4 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    _mm256_storeu_si256((__m256i*)buffer, v8);
    for (size_t i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += 8 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    return 0;
}


#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
__m256i STORM_popcnt256(__m256i v) {
    __m256i lookup1 = _mm256_setr_epi8(
        4, 5, 5, 6, 5, 6, 6, 7,
        5, 6, 6, 7, 6, 7, 7, 8,
        4, 5, 5, 6, 5, 6, 6, 7,
        5, 6, 6, 7, 6, 7, 7, 8
    );

    __m256i lookup2 = _mm256_setr_epi8(
        4, 3, 3, 2, 3, 2, 2, 1,
        3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1,
        3, 2, 2, 1, 2, 1, 1, 0
    );

    __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i popcnt1 = _mm256_shuffle_epi8(lookup1, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup2, hi);

    return _mm256_sad_epu8(popcnt1, popcnt2);
}

// modified from https://github.com/WojciechMula/sse-popcount
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static
uint64_t STORM_intersect_count_lookup_avx2_func(const uint8_t* STORM_RESTRICT data1, 
                                                const uint8_t* STORM_RESTRICT data2, 
                                                const size_t n)
{

    size_t i = 0;

    const __m256i lookup = _mm256_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i acc = _mm256_setzero_si256();

#define ITER { \
        const __m256i vec = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(data1 + i)), \
            _mm256_loadu_si256((const __m256i*)(data2 + i))); \
        const __m256i lo  = _mm256_and_si256(vec, low_mask); \
        const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask); \
        const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo); \
        const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi); \
        local = _mm256_add_epi8(local, popcnt1); \
        local = _mm256_add_epi8(local, popcnt2); \
        i += 32; \
    }

    while (i + 8*32 <= n) {
        __m256i local = _mm256_setzero_si256();
        ITER ITER ITER ITER
        ITER ITER ITER ITER
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
    }

    __m256i local = _mm256_setzero_si256();

    while (i + 32 <= n) {
        ITER;
    }

    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));

#undef ITER

    uint64_t result = 0;

    result += (uint64_t)(_mm256_extract_epi64(acc, 0));
    result += (uint64_t)(_mm256_extract_epi64(acc, 1));
    result += (uint64_t)(_mm256_extract_epi64(acc, 2));
    result += (uint64_t)(_mm256_extract_epi64(acc, 3));

    for (/**/; i < n; ++i) {
        result += STORM_popcnt_lookup8bit[data1[i] & data2[i]];
    }

    return result;
}

// modified from https://github.com/WojciechMula/sse-popcount
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static
uint64_t STORM_union_count_lookup_avx2_func(const uint8_t* STORM_RESTRICT data1, 
                                            const uint8_t* STORM_RESTRICT data2, 
                                            const size_t n)
    {

    size_t i = 0;

    const __m256i lookup = _mm256_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i acc = _mm256_setzero_si256();

#define ITER { \
        const __m256i vec = _mm256_or_si256(_mm256_loadu_si256((const __m256i*)(data1 + i)), \
            _mm256_loadu_si256((const __m256i*)(data2 + i))); \
        const __m256i lo  = _mm256_and_si256(vec, low_mask); \
        const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask); \
        const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo); \
        const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi); \
        local = _mm256_add_epi8(local, popcnt1); \
        local = _mm256_add_epi8(local, popcnt2); \
        i += 32; \
    }

    while (i + 8*32 <= n) {
        __m256i local = _mm256_setzero_si256();
        ITER ITER ITER ITER
        ITER ITER ITER ITER
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
    }

    __m256i local = _mm256_setzero_si256();

    while (i + 32 <= n) {
        ITER;
    }

    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));

#undef ITER

    uint64_t result = 0;

    result += (uint64_t)(_mm256_extract_epi64(acc, 0));
    result += (uint64_t)(_mm256_extract_epi64(acc, 1));
    result += (uint64_t)(_mm256_extract_epi64(acc, 2));
    result += (uint64_t)(_mm256_extract_epi64(acc, 3));

    for (/**/; i < n; ++i) {
        result += STORM_popcnt_lookup8bit[data1[i] | data2[i]];
    }

    return result;
}

// modified from https://github.com/WojciechMula/sse-popcount
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static
uint64_t STORM_diff_count_lookup_avx2_func(const uint8_t* STORM_RESTRICT data1, 
                                           const uint8_t* STORM_RESTRICT data2, 
                                           const size_t n)
{

    size_t i = 0;

    const __m256i lookup = _mm256_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i acc = _mm256_setzero_si256();

#define ITER { \
        const __m256i vec = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(data1 + i)), \
            _mm256_loadu_si256((const __m256i*)(data2 + i))); \
        const __m256i lo  = _mm256_and_si256(vec, low_mask); \
        const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask); \
        const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo); \
        const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi); \
        local = _mm256_add_epi8(local, popcnt1); \
        local = _mm256_add_epi8(local, popcnt2); \
        i += 32; \
    }

    while (i + 8*32 <= n) {
        __m256i local = _mm256_setzero_si256();
        ITER ITER ITER ITER
        ITER ITER ITER ITER
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
    }

    __m256i local = _mm256_setzero_si256();

    while (i + 32 <= n) {
        ITER;
    }

    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));

#undef ITER

    uint64_t result = 0;

    result += (uint64_t)(_mm256_extract_epi64(acc, 0));
    result += (uint64_t)(_mm256_extract_epi64(acc, 1));
    result += (uint64_t)(_mm256_extract_epi64(acc, 2));
    result += (uint64_t)(_mm256_extract_epi64(acc, 3));

    for (/**/; i < n; ++i) {
        result += STORM_popcnt_lookup8bit[data1[i] ^ data2[i]];
    }

    return result;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static
uint64_t STORM_popcnt_avx2(const __m256i* data, uint64_t size)
{
    __m256i cnt      = _mm256_setzero_si256();
    __m256i ones     = _mm256_setzero_si256();
    __m256i twos     = _mm256_setzero_si256();
    __m256i fours    = _mm256_setzero_si256();
    __m256i eights   = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

#define LOAD(a) (_mm256_loadu_si256(&data[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA256(&twosA, &ones, ones, LOAD(0), LOAD(1));
        STORM_CSA256(&twosB, &ones, ones, LOAD(2), LOAD(3));
        STORM_CSA256(&foursA, &twos, twos, twosA, twosB);
        STORM_CSA256(&twosA, &ones, ones, LOAD(4), LOAD(5));
        STORM_CSA256(&twosB, &ones, ones, LOAD(6), LOAD(7));
        STORM_CSA256(&foursB, &twos, twos, twosA, twosB);
        STORM_CSA256(&eightsA, &fours, fours, foursA, foursB);
        STORM_CSA256(&twosA, &ones, ones, LOAD(8), LOAD(9));
        STORM_CSA256(&twosB, &ones, ones, LOAD(10), LOAD(11));
        STORM_CSA256(&foursA, &twos, twos, twosA, twosB);
        STORM_CSA256(&twosA, &ones, ones, LOAD(12), LOAD(13));
        STORM_CSA256(&twosB, &ones, ones, LOAD(14), LOAD(15));
        STORM_CSA256(&foursB, &twos, twos, twosA, twosB);
        STORM_CSA256(&eightsB, &fours, fours, foursA, foursB);
        STORM_CSA256(&sixteens, &eights, eights, eightsA, eightsB);

        cnt = _mm256_add_epi64(cnt, STORM_popcnt256(sixteens));
    }
#undef LOAD

    cnt = _mm256_slli_epi64(cnt, 4);
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(eights), 3));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(fours), 2));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(twos), 1));
    cnt = _mm256_add_epi64(cnt, STORM_popcnt256(ones));

    for (/**/; i < size; ++i)
        cnt = _mm256_add_epi64(cnt, STORM_popcnt256(data[i]));

    cnt64 = (uint64_t*) &cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3];
}


/*
 * AVX2 Harley-Seal popcount (4th iteration).
 * The algorithm is based on the paper "Faster Population Counts
 * using AVX2 Instructions" by Daniel Lemire, Nathan Kurz and
 * Wojciech Mula (23 Nov 2016).
 * @see https://arxiv.org/abs/1611.07612
 */
// In this version we perform the operation A&B as input into the CSA operator.
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t STORM_intersect_count_csa_avx2(const __m256i* STORM_RESTRICT data1, 
                                        const __m256i* STORM_RESTRICT data2, 
                                        uint64_t size)
{
    __m256i cnt      = _mm256_setzero_si256();
    __m256i ones     = _mm256_setzero_si256();
    __m256i twos     = _mm256_setzero_si256();
    __m256i fours    = _mm256_setzero_si256();
    __m256i eights   = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

#define LOAD(a) (_mm256_loadu_si256(&data1[i+a]) & _mm256_loadu_si256(&data2[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA256(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA256(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA256(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA256(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA256(&sixteens,&eights, eights,eightsA,eightsB);

        cnt = _mm256_add_epi64(cnt, STORM_popcnt256(sixteens));
    }
#undef LOAD

    cnt = _mm256_slli_epi64(cnt, 4);
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(eights), 3));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(fours),  2));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(twos),   1));
    cnt = _mm256_add_epi64(cnt, STORM_popcnt256(ones));

    for (/**/; i < size; ++i)
        cnt = _mm256_add_epi64(cnt, STORM_popcnt256(_mm256_loadu_si256(&data1[i]) & _mm256_loadu_si256(&data2[i])));

    cnt64 = (uint64_t*) &cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3];
}

// In this version we perform the operation A|B as input into the CSA operator.
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t STORM_union_count_csa_avx2(const __m256i* STORM_RESTRICT data1, 
                                    const __m256i* STORM_RESTRICT data2, 
                                    uint64_t size)
{
    __m256i cnt      = _mm256_setzero_si256();
    __m256i ones     = _mm256_setzero_si256();
    __m256i twos     = _mm256_setzero_si256();
    __m256i fours    = _mm256_setzero_si256();
    __m256i eights   = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

#define LOAD(a) (_mm256_loadu_si256(&data1[i+a]) | _mm256_loadu_si256(&data2[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA256(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA256(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA256(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA256(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA256(&sixteens,&eights, eights,eightsA,eightsB);

        cnt = _mm256_add_epi64(cnt, STORM_popcnt256(sixteens));
    }
#undef LOAD

    cnt = _mm256_slli_epi64(cnt, 4);
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(eights), 3));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(fours),  2));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(twos),   1));
    cnt = _mm256_add_epi64(cnt, STORM_popcnt256(ones));

    for (/**/; i < size; ++i)
        cnt = _mm256_add_epi64(cnt, STORM_popcnt256(_mm256_loadu_si256(&data1[i]) | _mm256_loadu_si256(&data2[i])));

    cnt64 = (uint64_t*) &cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3];
}

// In this version we perform the operation A^B as input into the CSA operator.
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t STORM_diff_count_csa_avx2(const __m256i* STORM_RESTRICT data1, 
                                   const __m256i* STORM_RESTRICT data2, 
                                   uint64_t size)
{
    __m256i cnt      = _mm256_setzero_si256();
    __m256i ones     = _mm256_setzero_si256();
    __m256i twos     = _mm256_setzero_si256();
    __m256i fours    = _mm256_setzero_si256();
    __m256i eights   = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

#define LOAD(a) (_mm256_loadu_si256(&data1[i+a]) ^ _mm256_loadu_si256(&data2[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA256(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA256(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA256(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA256(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA256(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA256(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA256(&sixteens,&eights, eights,eightsA,eightsB);

        cnt = _mm256_add_epi64(cnt, STORM_popcnt256(sixteens));
    }
#undef LOAD

    cnt = _mm256_slli_epi64(cnt, 4);
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(eights), 3));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(fours),  2));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(STORM_popcnt256(twos),   1));
    cnt = _mm256_add_epi64(cnt, STORM_popcnt256(ones));

    for (/**/; i < size; ++i)
        cnt = _mm256_add_epi64(cnt, STORM_popcnt256(_mm256_loadu_si256(&data1[i]) ^ _mm256_loadu_si256(&data2[i])));

    cnt64 = (uint64_t*) &cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3];
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t STORM_intersect_count_avx2(const uint64_t* STORM_RESTRICT b1, 
                                    const uint64_t* STORM_RESTRICT b2, 
                                    const uint32_t n_ints)
{
    uint64_t count = 0;
    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;
    const uint32_t n_cycles = n_ints / 4;

    count += STORM_intersect_count_csa_avx2(r1, r2, n_cycles);

    for (int i = n_cycles*4; i < n_ints; ++i) {
        count += STORM_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t STORM_union_count_avx2(const uint64_t* STORM_RESTRICT b1, 
                   const uint64_t* STORM_RESTRICT b2, 
                   const uint32_t n_ints)
{
    uint64_t count = 0;
    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;
    const uint32_t n_cycles = n_ints / 4;

    count += STORM_union_count_csa_avx2(r1, r2, n_cycles);

    for (int i = n_cycles*4; i < n_ints; ++i) {
        count += STORM_POPCOUNT(b1[i] | b2[i]);
    }

    return(count);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t STORM_diff_count_avx2(const uint64_t* STORM_RESTRICT b1, 
                   const uint64_t* STORM_RESTRICT b2, 
                   const uint32_t n_ints)
{
    uint64_t count = 0;
    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;
    const uint32_t n_cycles = n_ints / 4;

    count += STORM_diff_count_csa_avx2(r1, r2, n_cycles);

    for (int i = n_cycles*4; i < n_ints; ++i) {
        count += STORM_POPCOUNT(b1[i] ^ b2[i]);
    }

    return(count);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t STORM_intersect_count_lookup_avx2(const uint64_t* STORM_RESTRICT b1, 
                                           const uint64_t* STORM_RESTRICT b2, 
                                           const uint32_t n_ints)
{
    return STORM_intersect_count_lookup_avx2_func((uint8_t*)b1, (uint8_t*)b2, n_ints*8);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t STORM_union_count_lookup_avx2(const uint64_t* STORM_RESTRICT b1, 
                                       const uint64_t* STORM_RESTRICT b2, 
                                       const uint32_t n_ints)
{
    return STORM_union_count_lookup_avx2_func((uint8_t*)b1, (uint8_t*)b2, n_ints*8);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t STORM_diff_count_lookup_avx2(const uint64_t* STORM_RESTRICT b1, 
                                      const uint64_t* STORM_RESTRICT b2, 
                                      const uint32_t n_ints)
{
    return STORM_diff_count_lookup_avx2_func((uint8_t*)b1, (uint8_t*)b2, n_ints*8);
}
#endif

/****************************
*  AVX512BW functions
****************************/

#if defined(HAVE_AVX512)

#include <immintrin.h>

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
STORM_FORCE_INLINE  
__m512i STORM_popcnt512(__m512i v) {
    __m512i m1 = _mm512_set1_epi8(0x55);
    __m512i m2 = _mm512_set1_epi8(0x33);
    __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512i t1 = _mm512_sub_epi8(v,       (_mm512_srli_epi16(v, 1)   & m1));
    __m512i t2 = _mm512_add_epi8(t1 & m2, (_mm512_srli_epi16(t1, 2)  & m2));
    __m512i t3 = _mm512_add_epi8(t2,       _mm512_srli_epi16(t2, 4)) & m4;

    return _mm512_sad_epu8(t3, _mm512_setzero_si512());
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
STORM_FORCE_INLINE  
void STORM_CSA512(__m512i* h, __m512i* l, __m512i a, __m512i b, __m512i c) {
    *l = _mm512_ternarylogic_epi32(c, b, a, 0x96);
    *h = _mm512_ternarylogic_epi32(c, b, a, 0xe8);
}

// By Wojciech Muła
// @see https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx512-harley-seal.cpp#L3
// @see https://arxiv.org/abs/1611.07612
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
STORM_FORCE_INLINE
__m512i STORM_avx512_popcount(const __m512i v) {
    const __m512i m1 = _mm512_set1_epi8(0x55); // 01010101
    const __m512i m2 = _mm512_set1_epi8(0x33); // 00110011
    const __m512i m4 = _mm512_set1_epi8(0x0F); // 00001111

    const __m512i t1 = _mm512_sub_epi8(v,       (_mm512_srli_epi16(v,  1)  & m1));
    const __m512i t2 = _mm512_add_epi8(t1 & m2, (_mm512_srli_epi16(t1, 2)  & m2));
    const __m512i t3 = _mm512_add_epi8(t2,       _mm512_srli_epi16(t2, 4)) & m4;
    return _mm512_sad_epu8(t3, _mm512_setzero_si512());
}

// 512i-version of carry-save adder subroutine.
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
STORM_FORCE_INLINE
void STORM_pospopcnt_csa_avx512(__m512i* STORM_RESTRICT h, 
                                __m512i* STORM_RESTRICT l, 
                                __m512i b, __m512i c) 
{
     *h = _mm512_ternarylogic_epi32(c, b, *l, 0xE8); // 11101000
     *l = _mm512_ternarylogic_epi32(c, b, *l, 0x96); // 10010110
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
uint64_t STORM_popcnt_avx512(const __m512i* STORM_RESTRICT data, uint64_t size)
{
    __m512i cnt      = _mm512_setzero_si512();
    __m512i ones     = _mm512_setzero_si512();
    __m512i twos     = _mm512_setzero_si512();
    __m512i fours    = _mm512_setzero_si512();
    __m512i eights   = _mm512_setzero_si512();
    __m512i sixteens = _mm512_setzero_si512();
    __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

#define LOAD(a) (_mm512_loadu_si512(&data[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA512(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA512(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA512(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA512(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA512(&sixteens,&eights, eights,eightsA,eightsB);

        cnt = _mm512_add_epi64(cnt, STORM_popcnt512(sixteens));
    }
#undef LOAD

    cnt = _mm512_slli_epi64(cnt, 4);
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(eights), 3));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(fours), 2));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(twos), 1));
    cnt = _mm512_add_epi64(cnt,  STORM_popcnt512(ones));

    for (/**/; i < size; ++i)
        cnt = _mm512_add_epi64(cnt, STORM_popcnt512(_mm512_loadu_si512(&data[i])));

    cnt64 = (uint64_t*)&cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3] +
            cnt64[4] +
            cnt64[5] +
            cnt64[6] +
            cnt64[7];
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static
int STORM_pospopcnt_u16_avx512bw_harvey_seal(const uint16_t* array, uint32_t len, uint32_t* flags) {
    for (uint32_t i = len - (len % (32 * 16)); i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

    const __m512i* data = (const __m512i*)array;
    __m512i v1  = _mm512_setzero_si512();
    __m512i v2  = _mm512_setzero_si512();
    __m512i v4  = _mm512_setzero_si512();
    __m512i v8  = _mm512_setzero_si512();
    __m512i v16 = _mm512_setzero_si512();
    __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;
    __m512i one = _mm512_set1_epi16(1);
    __m512i counter[16];

    const size_t size = len / 32;
    const uint64_t limit = size - size % 16;

    uint16_t buffer[32];

    uint64_t i = 0;
    while (i < limit) {
        for (size_t i = 0; i < 16; ++i)
            counter[i] = _mm512_setzero_si512();

        size_t thislimit = limit;
        if (thislimit - i >= (1 << 16))
            thislimit = i + (1 << 16) - 1;

        for (/**/; i < thislimit; i += 16) {
#define U(pos) {                     \
    counter[pos] = _mm512_add_epi16(counter[pos], _mm512_and_si512(v16, _mm512_set1_epi16(1))); \
    v16 = _mm512_srli_epi16(v16, 1); \
}
            STORM_pospopcnt_csa_avx512(&twosA,   &v1, _mm512_loadu_si512(data + i + 0), _mm512_loadu_si512(data + i + 1));
            STORM_pospopcnt_csa_avx512(&twosB,   &v1, _mm512_loadu_si512(data + i + 2), _mm512_loadu_si512(data + i + 3));
            STORM_pospopcnt_csa_avx512(&foursA,  &v2, twosA, twosB);
            STORM_pospopcnt_csa_avx512(&twosA,   &v1, _mm512_loadu_si512(data + i + 4), _mm512_loadu_si512(data + i + 5));
            STORM_pospopcnt_csa_avx512(&twosB,   &v1, _mm512_loadu_si512(data + i + 6), _mm512_loadu_si512(data + i + 7));
            STORM_pospopcnt_csa_avx512(&foursB,  &v2, twosA, twosB);
            STORM_pospopcnt_csa_avx512(&eightsA, &v4, foursA, foursB);
            STORM_pospopcnt_csa_avx512(&twosA,   &v1, _mm512_loadu_si512(data + i + 8),  _mm512_loadu_si512(data + i + 9));
            STORM_pospopcnt_csa_avx512(&twosB,   &v1, _mm512_loadu_si512(data + i + 10), _mm512_loadu_si512(data + i + 11));
            STORM_pospopcnt_csa_avx512(&foursA,  &v2, twosA, twosB);
            STORM_pospopcnt_csa_avx512(&twosA,   &v1, _mm512_loadu_si512(data + i + 12), _mm512_loadu_si512(data + i + 13));
            STORM_pospopcnt_csa_avx512(&twosB,   &v1, _mm512_loadu_si512(data + i + 14), _mm512_loadu_si512(data + i + 15));
            STORM_pospopcnt_csa_avx512(&foursB,  &v2, twosA, twosB);
            STORM_pospopcnt_csa_avx512(&eightsB, &v4, foursA, foursB);
            U(0) U(1) U(2) U(3) U(4) U(5) U(6) U(7) U(8) U(9) U(10) U(11) U(12) U(13) U(14) U(15) // Updates
            STORM_pospopcnt_csa_avx512(&v16,     &v8, eightsA, eightsB);
        }
        // Update the counters after the last iteration.
        for (size_t i = 0; i < 16; ++i) U(i)
#undef U
        
        for (size_t i = 0; i < 16; ++i) {
            _mm512_storeu_si512((__m512i*)buffer, counter[i]);
            for (size_t z = 0; z < 32; z++) {
                flags[i] += 16 * (uint32_t)buffer[z];
            }
        }
    }

    _mm512_storeu_si512((__m512i*)buffer, v1);
    for (size_t i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 1 * ((buffer[i] & (1 << j)) >> j);
        }
    }

    _mm512_storeu_si512((__m512i*)buffer, v2);
    for (size_t i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 2 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    
    _mm512_storeu_si512((__m512i*)buffer, v4);
    for (size_t i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 4 * ((buffer[i] & (1 << j)) >> j);
        }
    }

    _mm512_storeu_si512((__m512i*)buffer, v8);
    for (size_t i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 8 * ((buffer[i] & (1 << j)) >> j);
        }
    }
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
int STORM_pospopcnt_u16_avx512bw_blend_popcnt_unroll8(const uint16_t* data, uint32_t len, uint32_t* flags) { 
#define AND_OR 0xea // ternary function: (a & b) | c
    const __m512i* data_vectors = (const __m512i*)(data);
    const uint32_t n_cycles = len / 32;

    size_t i = 0;
    for (/**/; i + 8 <= n_cycles; i += 8) {
#define L(p) __m512i v##p = _mm512_loadu_si512(data_vectors+i+p);
        L(0)  L(1)  L(2)  L(3)  
        L(4)  L(5)  L(6)  L(7) 

#define U0(p,k) __m512i input##p = _mm512_ternarylogic_epi32(v##p, _mm512_set1_epi16(0x00FF), _mm512_slli_epi16(v##k, 8), AND_OR);
#define U1(p,k) __m512i input##k = _mm512_ternarylogic_epi32(v##p, _mm512_set1_epi16(0xFF00), _mm512_srli_epi16(v##k, 8), AND_OR);
#define U(p, k)  U0(p,k) U1(p,k)

        U(0,1) U( 2, 3) U( 4, 5) U( 6, 7)
        
        for (int i = 0; i < 8; ++i) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u64(_mm512_movepi8_mask(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u64(_mm512_movepi8_mask(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A(2, 3) A(4,5) A(6, 7)

#define P0(p) input##p = _mm512_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)

            P(0,1) P(2, 3) P(4,5) P(6, 7)
        }
    }

    for (/**/; i + 4 <= n_cycles; i += 4) {
        L(0) L(1) L(2) L(3)
        U(0,1) U(2,3)
        
        for (int i = 0; i < 8; ++i) {
            A(0,1) A(2, 3)
            P(0,1) P(2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; ++i) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 32;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((data[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P
#undef AND_OR
    
    return 0;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static
int STORM_pospopcnt_u16_avx512bw_adder_forest(const uint16_t* array, uint32_t len, uint32_t* flags) {
    __m512i counters[16];

    for (size_t i = 0; i < 16; ++i) {
        counters[i] = _mm512_setzero_si512();
    }

    const __m512i mask1bit = _mm512_set1_epi16(0x5555); // 0101010101010101 Pattern: 01
    const __m512i mask2bit = _mm512_set1_epi16(0x3333); // 0011001100110011 Pattern: 0011
    const __m512i mask4bit = _mm512_set1_epi16(0x0F0F); // 0000111100001111 Pattern: 00001111
    const __m512i mask8bit = _mm512_set1_epi16(0x00FF); // 0000000011111111 Pattern: 0000000011111111
    
    const uint32_t n_cycles = len / (2048 * (16*32));
    const uint32_t n_total  = len / (16*32);
    uint16_t tmp[32];

/*------ Macros --------*/
#define LE(i,p,k)  const __m512i sum##p##k##_##i##bit_even = _mm512_add_epi8(input##p & mask##i##bit, input##k & mask##i##bit);
#define LO(i,p,k)  const __m512i sum##p##k##_##i##bit_odd  = _mm512_add_epi8(_mm512_srli_epi16(input##p, i) & mask##i##bit, _mm512_srli_epi16(input##k, i) & mask##i##bit);

#define LBLOCK(i)           \
    LE(i,0,1)   LO(i,0,1)   \
    LE(i,2,3)   LO(i,2,3)   \
    LE(i,4,5)   LO(i,4,5)   \
    LE(i,6,7)   LO(i,6,7)   \
    LE(i,8,9)   LO(i,8,9)   \
    LE(i,10,11) LO(i,10,11) \
    LE(i,12,13) LO(i,12,13) \
    LE(i,14,15) LO(i,14,15) \

#define EVEN(b,i,k,p) input##i = sum##k##p##_##b##bit_even;
#define ODD(b,i,k,p)  input##i = sum##k##p##_##b##bit_odd;

#define UPDATE(i)                                                  \
    EVEN(i,0,0,1) EVEN(i,1,2,3)   EVEN(i,2,4,5)   EVEN(i,3,6,7)    \
    EVEN(i,4,8,9) EVEN(i,5,10,11) EVEN(i,6,12,13) EVEN(i,7,14,15)  \
     ODD(i,8,0,1)  ODD(i,9,2,3)    ODD(i,10,4,5)   ODD(i,11,6,7)   \
     ODD(i,12,8,9) ODD(i,13,10,11) ODD(i,14,12,13) ODD(i,15,14,15) \

#define UE(i,p,k) counters[i] = _mm512_add_epi16(counters[i], sum##p##k##_8bit_even);
#define UO(i,p,k) counters[i] = _mm512_add_epi16(counters[i], sum##p##k##_8bit_odd);

/*------ Start --------*/
#define L(p) __m512i input##p = _mm512_loadu_si512((__m512i*)(array + i*2048*512 + j*512 + p*32));
    size_t i = 0;
    for (/**/; i < n_cycles; ++i) {
        for (int j = 0; j < 2048; ++j) {
            // Load 16 registers.
            L(0)  L(1)  L(2)  L(3)  
            L(4)  L(5)  L(6)  L(7) 
            L(8)  L(9)  L(10) L(11) 
            L(12) L(13) L(14) L(15)

            // Perform updates for bits {1,2,4,8}.
            LBLOCK(1) UPDATE(1)
            LBLOCK(2) UPDATE(2)
            LBLOCK(4) UPDATE(4)
            LBLOCK(8) UPDATE(8)

            // Update accumulators.
            UE( 0,0,1) UE( 1, 2, 3) UE( 2, 4, 5) UE( 3, 6, 7)  
            UE( 4,8,9) UE( 5,10,11) UE( 6,12,13) UE( 7,14,15) 
            UO( 8,0,1) UO( 9, 2, 3) UO(10, 4, 5) UO(11, 6, 7) 
            UO(12,8,9) UO(13,10,11) UO(14,12,13) UO(15,14,15)
        }

        // Update.
        for (size_t i = 0; i < 16; ++i) {
            _mm512_storeu_si512((__m512i*)tmp, counters[i]);
            for (int j = 0; j < 32; ++j) flags[i] += tmp[j];
        }
        // Reset.
        for (size_t i = 0; i < 16; ++i) {
            counters[i] = _mm512_setzero_si512();
        }
    }
#undef L
#define L(p) __m512i input##p = _mm512_loadu_si512((__m512i*)(array + i*512 + p*32));
    i *= 2048;
    for (/**/; i < n_total; ++i) {
        // Load 16 registers.
        L(0)  L(1)  L(2)  L(3)  
        L(4)  L(5)  L(6)  L(7) 
        L(8)  L(9)  L(10) L(11) 
        L(12) L(13) L(14) L(15)

        // Perform updates for bits {1,2,4,8}.
        LBLOCK(1) UPDATE(1)
        LBLOCK(2) UPDATE(2)
        LBLOCK(4) UPDATE(4)
        LBLOCK(8) UPDATE(8)

        // Update accumulators.
        UE( 0,0,1) UE( 1, 2, 3) UE( 2, 4, 5) UE( 3, 6, 7)  
        UE( 4,8,9) UE( 5,10,11) UE( 6,12,13) UE( 7,14,15) 
        UO( 8,0,1) UO( 9, 2, 3) UO(10, 4, 5) UO(11, 6, 7) 
        UO(12,8,9) UO(13,10,11) UO(14,12,13) UO(15,14,15)
    }

    i *= 512;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef UPDATE
#undef ODD
#undef EVEN
#undef LBLOCK
#undef LE
#undef LO
#undef UO
#undef UE

    for (size_t i = 0; i < 16; ++i) {
        _mm512_storeu_si512((__m512i*)tmp, counters[i]);
        for (int j = 0; j < 32; ++j) flags[i] += tmp[j];
    }
    return 0;
}

/*
 * AVX512 Harley-Seal popcount (4th iteration).
 * The algorithm is based on the paper "Faster Population Counts
 * using AVX2 Instructions" by Daniel Lemire, Nathan Kurz and
 * Wojciech Mula (23 Nov 2016).
 * @see https://arxiv.org/abs/1611.07612
 */
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
uint64_t STORM_intersect_count_csa_avx512(const __m512i* STORM_RESTRICT data1, 
                                          const __m512i* STORM_RESTRICT data2, 
                                          uint64_t size)
{
    __m512i cnt      = _mm512_setzero_si512();
    __m512i ones     = _mm512_setzero_si512();
    __m512i twos     = _mm512_setzero_si512();
    __m512i fours    = _mm512_setzero_si512();
    __m512i eights   = _mm512_setzero_si512();
    __m512i sixteens = _mm512_setzero_si512();
    __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

#define LOAD(a) (_mm512_loadu_si512(&data1[i+a]) & _mm512_loadu_si512(&data2[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA512(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA512(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA512(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA512(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA512(&sixteens,&eights, eights,eightsA,eightsB);

        cnt = _mm512_add_epi64(cnt, STORM_popcnt512(sixteens));
    }
#undef LOAD

    cnt = _mm512_slli_epi64(cnt, 4);
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(eights), 3));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(fours), 2));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(twos), 1));
    cnt = _mm512_add_epi64(cnt,  STORM_popcnt512(ones));

    for (/**/; i < size; ++i)
        cnt = _mm512_add_epi64(cnt, STORM_popcnt512(_mm512_loadu_si512(&data1[i]) & _mm512_loadu_si512(&data2[i])));

    cnt64 = (uint64_t*)&cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3] +
            cnt64[4] +
            cnt64[5] +
            cnt64[6] +
            cnt64[7];
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
uint64_t STORM_union_count_csa_avx512(const __m512i* STORM_RESTRICT data1, 
                                      const __m512i* STORM_RESTRICT data2, 
                                      uint64_t size)
{
    __m512i cnt      = _mm512_setzero_si512();
    __m512i ones     = _mm512_setzero_si512();
    __m512i twos     = _mm512_setzero_si512();
    __m512i fours    = _mm512_setzero_si512();
    __m512i eights   = _mm512_setzero_si512();
    __m512i sixteens = _mm512_setzero_si512();
    __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

#define LOAD(a) (_mm512_loadu_si512(&data1[i+a]) | _mm512_loadu_si512(&data2[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA512(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA512(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA512(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA512(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA512(&sixteens,&eights, eights,eightsA,eightsB);

        cnt = _mm512_add_epi64(cnt, STORM_popcnt512(sixteens));
    }
#undef LOAD

    cnt = _mm512_slli_epi64(cnt, 4);
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(eights), 3));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(fours), 2));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(twos), 1));
    cnt = _mm512_add_epi64(cnt,  STORM_popcnt512(ones));

    for (/**/; i < size; ++i)
        cnt = _mm512_add_epi64(cnt, STORM_popcnt512(_mm512_loadu_si512(&data1[i]) | _mm512_loadu_si512(&data2[i])));

    cnt64 = (uint64_t*)&cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3] +
            cnt64[4] +
            cnt64[5] +
            cnt64[6] +
            cnt64[7];
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
uint64_t STORM_diff_count_csa_avx512(const __m512i* STORM_RESTRICT data1, 
                                  const __m512i* STORM_RESTRICT data2, 
                                  uint64_t size)
{
    __m512i cnt      = _mm512_setzero_si512();
    __m512i ones     = _mm512_setzero_si512();
    __m512i twos     = _mm512_setzero_si512();
    __m512i fours    = _mm512_setzero_si512();
    __m512i eights   = _mm512_setzero_si512();
    __m512i sixteens = _mm512_setzero_si512();
    __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

#define LOAD(a) (_mm512_loadu_si512(&data1[i+a]) ^ _mm512_loadu_si512(&data2[i+a]))

    for (/**/; i < limit; i += 16) {
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(0), LOAD(1));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(2), LOAD(3));
        STORM_CSA512(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(4), LOAD(5));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(6), LOAD(7));
        STORM_CSA512(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&eightsA, &fours,  fours, foursA, foursB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(8), LOAD(9));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(10), LOAD(11));
        STORM_CSA512(&foursA,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&twosA,   &ones,   ones,  LOAD(12), LOAD(13));
        STORM_CSA512(&twosB,   &ones,   ones,  LOAD(14), LOAD(15));
        STORM_CSA512(&foursB,  &twos,   twos,  twosA,  twosB);
        STORM_CSA512(&eightsB, &fours,  fours, foursA, foursB);
        STORM_CSA512(&sixteens,&eights, eights,eightsA,eightsB);

        cnt = _mm512_add_epi64(cnt, STORM_popcnt512(sixteens));
    }
#undef LOAD

    cnt = _mm512_slli_epi64(cnt, 4);
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(eights), 3));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(fours), 2));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(STORM_popcnt512(twos), 1));
    cnt = _mm512_add_epi64(cnt,  STORM_popcnt512(ones));

    for (/**/; i < size; ++i)
        cnt = _mm512_add_epi64(cnt, STORM_popcnt512(_mm512_loadu_si512(&data1[i]) ^ _mm512_loadu_si512(&data2[i])));

    cnt64 = (uint64_t*)&cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3] +
            cnt64[4] +
            cnt64[5] +
            cnt64[6] +
            cnt64[7];
}

// Functions
// AVX512
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
uint64_t STORM_intersect_count_avx512(const uint64_t* STORM_RESTRICT b1, 
                                      const uint64_t* STORM_RESTRICT b2, 
                                      const uint32_t n_ints) 
{
    uint64_t count = 0;
    const __m512i* r1 = (const __m512i*)(b1);
    const __m512i* r2 = (const __m512i*)(b2);
    const uint32_t n_cycles = n_ints / 8;

    count += STORM_intersect_count_csa_avx512(r1, r2, n_cycles);

    for (int i = n_cycles*8; i < n_ints; ++i) {
        count += STORM_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
uint64_t STORM_union_count_avx512(const uint64_t* STORM_RESTRICT b1, 
                                  const uint64_t* STORM_RESTRICT b2, 
                                  const uint32_t n_ints) 
{
    uint64_t count = 0;
    const __m512i* r1 = (const __m512i*)(b1);
    const __m512i* r2 = (const __m512i*)(b2);
    const uint32_t n_cycles = n_ints / 8;

    count += STORM_union_count_csa_avx512(r1, r2, n_cycles);

    for (int i = n_cycles*8; i < n_ints; ++i) {
        count += STORM_POPCOUNT(b1[i] | b2[i]);
    }

    return(count);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
uint64_t STORM_diff_count_avx512(const uint64_t* STORM_RESTRICT b1, 
                                 const uint64_t* STORM_RESTRICT b2, 
                                 const uint32_t n_ints) 
{
    uint64_t count = 0;
    const __m512i* r1 = (const __m512i*)(b1);
    const __m512i* r2 = (const __m512i*)(b2);
    const uint32_t n_cycles = n_ints / 8;

    count += STORM_diff_count_csa_avx512(r1, r2, n_cycles);

    for (int i = n_cycles*8; i < n_ints; ++i) {
        count += STORM_POPCOUNT(b1[i] ^ b2[i]);
    }

    return(count);
}
#endif

/****************************
*  Popcount
****************************/

STORM_FORCE_INLINE
uint64_t STORM_popcount64_unrolled(const uint64_t* data, uint64_t size) {
    uint64_t i = 0;
    uint64_t limit = size - size % 4;
    uint64_t cnt = 0;

    for (/**/; i < limit; i += 4) {
        cnt += STORM_popcount64(data[i+0]);
        cnt += STORM_popcount64(data[i+1]);
        cnt += STORM_popcount64(data[i+2]);
        cnt += STORM_popcount64(data[i+3]);
    }

    for (/**/; i < size; ++i)
        cnt += STORM_popcount64(data[i]);

    return cnt;
}

/****************************
*  Scalar functions
****************************/

STORM_FORCE_INLINE 
uint64_t STORM_intersect_count_scalar(const uint64_t* STORM_RESTRICT b1, 
                                      const uint64_t* STORM_RESTRICT b2, 
                                      const uint32_t n_ints)
{
    return STORM_intersect_count_unrolled(b1, b2, n_ints);
}

STORM_FORCE_INLINE 
uint64_t STORM_union_count_scalar(const uint64_t* STORM_RESTRICT b1, 
                                  const uint64_t* STORM_RESTRICT b2, 
                                  const uint32_t n_ints)
{
    return STORM_union_count_unrolled(b1, b2, n_ints);
}

STORM_FORCE_INLINE 
uint64_t STORM_diff_count_scalar(const uint64_t* STORM_RESTRICT b1, 
                                 const uint64_t* STORM_RESTRICT b2, 
                                 const uint32_t n_ints)
{
    return STORM_diff_count_unrolled(b1, b2, n_ints);
}

static
uint64_t STORM_intersect_count_scalar_list(const uint64_t* STORM_RESTRICT b1, 
                                           const uint64_t* STORM_RESTRICT b2, 
                                           const uint32_t* STORM_RESTRICT l1, 
                                           const uint32_t* STORM_RESTRICT l2,
                                           const uint32_t n1, 
                                           const uint32_t n2) 
{
    uint64_t count = 0;

#define MOD(x) (( (x) * 64 ) >> 6)
    if(n1 < n2) {
        for (int i = 0; i < n1; ++i) {
            count += ((b2[l1[i] >> 6] & (1L << MOD(l1[i]))) != 0); 
            __builtin_prefetch(&b2[l1[i] >> 6], 0, _MM_HINT_T0);
        }
    } else {
        for (int i = 0; i < n2; ++i) {
            count += ((b1[l2[i] >> 6] & (1L << MOD(l2[i]))) != 0);
            __builtin_prefetch(&b1[l2[i] >> 6], 0, _MM_HINT_T0);
        }
    }
#undef MOD
    return(count);
}


/* *************************************
*  Function pointer definitions.
***************************************/
typedef uint64_t (*STORM_compute_func)(const uint64_t*, const uint64_t*, const uint32_t);
typedef uint64_t (*STORM_compute_lfunc)(const uint64_t*, const uint64_t*, 
    const uint32_t*, const uint32_t*, const uint32_t, const uint32_t);
typedef int (STORM_pposcnt_func)(const uint16_t*, uint32_t, uint32_t*);

/* *************************************
*  Alignment 
***************************************/
// Return the best alignment given the available instruction set at
// run-time.
static 
uint32_t STORM_get_alignment() {

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

        #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
        #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
        #endif
    }
    #endif
#endif

    uint32_t alignment = 0;
#if defined(HAVE_AVX512)
    if ((cpuid & STORM_bit_AVX512BW)) { // 16*512
        alignment = STORM_AVX512_ALIGNMENT;
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & STORM_bit_AVX2) && alignment == 0) { // 16*256
        alignment = STORM_AVX2_ALIGNMENT;
    }
#endif

#if defined(HAVE_SSE42)
    if ((cpuid & STORM_bit_SSE41) && alignment == 0) { // 16*128
        alignment = STORM_SSE_ALIGNMENT;
    }
#endif

    if (alignment == 0) alignment = 8;
    return alignment;
}

/* *************************************
*  Set algebra functions
***************************************/
// Return the optimal intersection function given the range [0, n_bitmaps_vector)
// and the available instruction set at run-time.
static
STORM_compute_func STORM_get_intersect_count_func(const uint32_t n_bitmaps_vector) {

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

        #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
        #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
        #endif
    }
    #endif
#endif


#if defined(HAVE_AVX512)
    if ((cpuid & STORM_bit_AVX512BW) && n_bitmaps_vector >= 128) { // 16*512
        return &STORM_intersect_count_avx512;
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & STORM_bit_AVX2) && n_bitmaps_vector >= 64) { // 16*256
        return &STORM_intersect_count_avx2;
    }
    
    if ((cpuid & STORM_bit_AVX2) && n_bitmaps_vector >= 4) {
        return &STORM_intersect_count_lookup_avx2;
    }
#endif

#if defined(HAVE_SSE42)
    if ((cpuid & STORM_bit_SSE41) && n_bitmaps_vector >= 32) { // 16*128
        return &STORM_intersect_count_sse4;
    }
#endif

    return &STORM_intersect_count_scalar;
}

static
STORM_compute_func STORM_get_union_count_func(const uint32_t n_bitmaps_vector) {

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

        #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
        #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
        #endif
    }
    #endif
#endif


#if defined(HAVE_AVX512)
    if ((cpuid & STORM_bit_AVX512BW) && n_bitmaps_vector >= 128) { // 16*512
        return &STORM_union_count_avx512;
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & STORM_bit_AVX2) && n_bitmaps_vector >= 64) { // 16*256
        return &STORM_union_count_avx2;
    }
    
    if ((cpuid & STORM_bit_AVX2) && n_bitmaps_vector >= 4) {
        return &STORM_union_count_lookup_avx2;
    }
#endif

#if defined(HAVE_SSE42)
    if ((cpuid & STORM_bit_SSE41) && n_bitmaps_vector >= 32) { // 16*128
        return &STORM_union_count_sse4;
    }
#endif

    return &STORM_union_count_scalar;
}

static
STORM_compute_func STORM_get_diff_count_func(const uint32_t n_bitmaps_vector) {

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

        #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
        #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
        #endif
    }
    #endif
#endif


#if defined(HAVE_AVX512)
    if ((cpuid & STORM_bit_AVX512BW) && n_bitmaps_vector >= 128) { // 16*512
        return &STORM_diff_count_avx512;
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & STORM_bit_AVX2) && n_bitmaps_vector >= 64) { // 16*256
        return &STORM_diff_count_avx2;
    }
    
    if ((cpuid & STORM_bit_AVX2) && n_bitmaps_vector >= 4) {
        return &STORM_diff_count_lookup_avx2;
    }
#endif

#if defined(HAVE_SSE42)
    if ((cpuid & STORM_bit_SSE41) && n_bitmaps_vector >= 32) { // 16*128
        return &STORM_diff_count_sse4;
    }
#endif

    return &STORM_diff_count_scalar;
}

// real
// Return the optimal intersection function given the range [0, n_bitmaps_vector)
// and the available instruction set at run-time.
static
uint64_t STORM_intersect_count(const uint64_t* STORM_RESTRICT data1, const uint64_t* STORM_RESTRICT data2, const uint32_t n_len) {

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

        #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
        #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
        #endif
    }
    #endif
#endif


#if defined(HAVE_AVX512)
    if ((cpuid & STORM_bit_AVX512BW) && n_len >= 128) { // 16*512
        return STORM_intersect_count_avx512(data1, data2, n_len);
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & STORM_bit_AVX2) && n_len >= 64) { // 16*256
        return STORM_intersect_count_avx2(data1, data2, n_len);
    }
    
    if ((cpuid & STORM_bit_AVX2) && n_len >= 4) {
        return STORM_intersect_count_lookup_avx2(data1, data2, n_len);
    }
#endif

#if defined(HAVE_SSE42)
    if ((cpuid & STORM_bit_SSE41) && n_len >= 32) { // 16*128
        return STORM_intersect_count_sse4(data1, data2, n_len);
    }
#endif

    return STORM_intersect_count_scalar(data1, data2, n_len);
}

static
uint64_t STORM_union_count(const uint64_t* STORM_RESTRICT data1, const uint64_t* STORM_RESTRICT data2, const uint32_t n_len) {

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

        #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
        #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
        #endif
    }
    #endif
#endif


#if defined(HAVE_AVX512)
    if ((cpuid & STORM_bit_AVX512BW) && n_len >= 128) { // 16*512
        return STORM_union_count_avx512(data1, data2, n_len);
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & STORM_bit_AVX2) && n_len >= 64) { // 16*256
        return STORM_union_count_avx2(data1, data2, n_len);
    }
    
    if ((cpuid & STORM_bit_AVX2) && n_len >= 4) {
        return STORM_union_count_lookup_avx2(data1, data2, n_len);
    }
#endif

#if defined(HAVE_SSE42)
    if ((cpuid & STORM_bit_SSE41) && n_len >= 32) { // 16*128
        return STORM_union_count_sse4(data1, data2, n_len);
    }
#endif

    return STORM_union_count_scalar(data1, data2, n_len);
}

static
uint64_t STORM_diff_count(const uint64_t* STORM_RESTRICT data1, const uint64_t* STORM_RESTRICT data2, const uint32_t n_len) {

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

        #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
        #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
        #endif
    }
    #endif
#endif


#if defined(HAVE_AVX512)
    if ((cpuid & STORM_bit_AVX512BW) && n_len >= 128) { // 16*512
        return STORM_diff_count_avx512(data1, data2, n_len);
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & STORM_bit_AVX2) && n_len >= 64) { // 16*256
        return STORM_diff_count_avx2(data1, data2, n_len);
    }
    
    if ((cpuid & STORM_bit_AVX2) && n_len >= 4) {
        return STORM_diff_count_lookup_avx2(data1, data2, n_len);
    }
#endif

#if defined(HAVE_SSE42)
    if ((cpuid & STORM_bit_SSE41) && n_len >= 32) { // 16*128
        return STORM_diff_count_sse4(data1, data2, n_len);
    }
#endif

    return STORM_diff_count_scalar(data1, data2, n_len);
}

/* *************************************
*  POPCNT and POSPOPCNT functions.
***************************************/
static
uint64_t STORM_popcnt(const uint8_t* data, uint32_t size) {
    uint64_t cnt = 0;
    uint64_t i;

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
        /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

    #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
    #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
    #endif
    }
    #endif
#endif

#if defined(HAVE_AVX512)

    /* AVX512 requires arrays >= 1024 bytes */
    if ((cpuid & STORM_bit_AVX512BW) &&
        size >= 1024)
    {
        cnt += STORM_popcnt_avx512((const __m512i*)data, size / 64);
        data += size - size % 64;
        size = size % 64;
    }

#endif

#if defined(HAVE_AVX2)

    /* AVX2 requires arrays >= 512 bytes */
    if ((cpuid & STORM_bit_AVX2) &&
        size >= 512)
    {
        cnt += STORM_popcnt_avx2((const __m256i*)data, size / 32);
        data += size - size % 32;
        size = size % 32;
    }

#endif

#if defined(HAVE_POPCNT)

    if (cpuid & STORM_bit_POPCNT) {
        cnt += STORM_popcount64_unrolled((const uint64_t*)data, size / 8);
        data += size - size % 8;
        size = size % 8;
        for (i = 0; i < size; i++)
            cnt += STORM_popcount64(data[i]);

        return cnt;
    }

#endif

    /* pure integer popcount algorithm */
    if (size >= 8) {
        cnt += STORM_popcount64_unrolled((const uint64_t*)data, size / 8);
        data += size - size % 8;
        size = size % 8;
    }

    /* pure integer popcount algorithm */
    for (i = 0; i < size; i++)
        cnt += STORM_popcount64(data[i]);

    return cnt;
}

static
int STORM_pospopcnt_u16(const uint16_t* data, uint32_t len, uint32_t* flags) {
    memset(flags, 0, sizeof(uint32_t)*16);

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
        /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

    #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
    #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
    #endif
    }
    #endif
#endif

#if defined(HAVE_AVX512)
    if ((cpuid & STORM_bit_AVX512BW))
    {
        if (len < 32) return(STORM_pospopcnt_u16_sse_sad(data, len, flags)); // small
        else if (len < 256)  return(STORM_pospopcnt_u16_sse_blend_popcnt_unroll8(data, len, flags)); // small
        else if (len < 512)  return(STORM_pospopcnt_u16_avx512bw_blend_popcnt_unroll8(data, len, flags)); // medium
        else if (len < 4096) return(STORM_pospopcnt_u16_avx512bw_adder_forest(data, len, flags)); // medium3
        else return(STORM_pospopcnt_u16_avx512bw_harvey_seal(data, len, flags)); // fix
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & STORM_bit_AVX2))
    {
        if (len < 128) return(STORM_pospopcnt_u16_sse_sad(data, len, flags)); // small
        else if (len < 1024) return(STORM_pospopcnt_u16_avx2_blend_popcnt_unroll8(data, len, flags)); // medium
        else return(STORM_pospopcnt_u16_avx2_harvey_seal(data, len, flags)); // large
    }
#endif

#if defined(HAVE_SSE4)
    if ((cpuid & STORM_bit_SSE42))
    {
         return(STORM_pospopcnt_u16_sse_harvey_seal(data, len, flags));
    }
#endif

#ifndef _MSC_VER
    return(STORM_pospopcnt_u16_scalar_umul128_unroll2(data, len, flags)); // fallback scalar
#else
    return(STORM_pospopcnt_u16_scalar_naive(data, len, flags));
#endif
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBALGEBRA_H_9827563662203 */
