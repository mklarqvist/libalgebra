#include "libalgebra.h"

#include <iostream>
#include <random>
#include <chrono>

int benchmark(uint32_t range, uint32_t n_values) {
    std::cout << "Range = [0," << range << ") with bits=" << n_values << std::endl;

    // PRNG
    std::uniform_int_distribution<uint32_t> distr(0, range-1); // right inclusive
    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    // Align some bitmaps.
    uint32_t n_bitmaps = ceil(range / 64.0);
    uint64_t* bitmaps  = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), n_bitmaps*sizeof(uint64_t));
    uint64_t* bitmaps2 = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), n_bitmaps*sizeof(uint64_t));
    
    // Generate some random data.
    uint32_t n_unique = 0;
    while (n_unique < n_values) {
        uint32_t val = distr(eng);
        n_unique += (bitmaps[val / 64] & (1ULL << (val % 64))) == 0;
        bitmaps[val / 64] |= 1ULL << (val % 64);
    }

    n_unique = 0;
    while (n_unique < n_values) {
        uint32_t val = distr(eng);
        n_unique += (bitmaps2[val / 64] & (1ULL << (val % 64))) == 0;
        bitmaps2[val / 64] |= 1ULL << (val % 64);
    }

    // Compute the population bit count.
    uint64_t bitcnt = STORM_popcnt((uint8_t*)bitmaps, n_bitmaps*8);
    std::cout << "POPCNT=" << bitcnt << std::endl;
    
    // Compute the positional bit count.
    uint32_t flag_count[16];
    int pospopcnt = STORM_pospopcnt_u16((uint16_t*)bitmaps, n_bitmaps*4, &flag_count[0]);
    uint32_t pospopcnt_total = flag_count[0];
    std::cout << "POSPOPCNT=" << flag_count[0];
    for (int i = 1; i < 16; ++i) {
        std::cout << "," << flag_count[i];
        pospopcnt_total += flag_count[i];
    }
    std::cout << std::endl;
    std::cout << "POSPOPCNT total=" << pospopcnt_total << std::endl;

    // Compute intersect count
    uint64_t intersect_count = STORM_intersect_count(bitmaps, bitmaps2, n_bitmaps);
    std::cout << "intersect count=" << intersect_count << std::endl;

    // Compute union count
    uint64_t union_count = STORM_union_count(bitmaps, bitmaps2, n_bitmaps);
    std::cout << "union count=" << union_count << std::endl;

    // Compute diff count
    uint64_t diff_count = STORM_diff_count(bitmaps, bitmaps2, n_bitmaps);
    std::cout << "diff count=" << diff_count << std::endl;

    // Clean up.
    STORM_aligned_free(bitmaps);
    STORM_aligned_free(bitmaps2);
    return 1;
}

int main(int argc, char **argv) { 
    benchmark(2048000, 50000);

    return EXIT_SUCCESS;
}