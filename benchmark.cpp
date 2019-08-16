#include "libalgebra.h"

#include <iostream>
#include <random>
#include <chrono>

struct bench_unit {
    bench_unit() : valid(false), cycles(0), cycles_local(0), times(0), times_local(0){}

    bool valid;
    float cycles;
    float cycles_local;
    uint64_t times;
    uint64_t times_local;
};

uint64_t get_cpu_cycles() {
    uint64_t result;
#ifndef _MSC_VER
    __asm__ volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax":"=a"
                     (result)::"%rdx");
#else
    result = __rdtsc();
#endif
    return result;
};

void generate_random_data(uint64_t* data, uint32_t range, uint32_t n) {
    // PRNG
    std::uniform_int_distribution<uint32_t> distr(0, range-1); // right inclusive
    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

     // Generate some random data.
    uint32_t n_unique = 0;
    // while (n_unique < n) {
    for (int i = 0; i < n; ++i) {
        uint32_t val = distr(eng);
        n_unique += (data[val / 64] & (1ULL << (val % 64))) == 0;
        data[val / 64] |= 1ULL << (val % 64);
    }
}

// Definition for microsecond timer.
typedef std::chrono::high_resolution_clock::time_point clockdef;

int set_algebra_wrapper(std::string name,
    STORM_compute_func f, 
    int iterations,
    uint64_t* STORM_RESTRICT data1, 
    uint64_t* STORM_RESTRICT data2, 
    uint32_t range,
    uint32_t n_values,
    uint32_t n_bitmaps, 
    bench_unit& unit) 
{
    uint32_t cycles_low = 0, cycles_high = 0;
    uint32_t cycles_low1 = 0, cycles_high1 = 0;
    // Start timer.

    std::vector<uint64_t> clocks;
    std::vector<uint32_t> times;

#ifndef _MSC_VER
// Intel guide:
// @see: https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/ia-32-ia-64-benchmark-code-execution-paper.pdf
asm   volatile ("CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "%rax", "%rbx", "%rcx", "%rdx"); 
asm   volatile("RDTSCP\n\t"
               "mov %%edx, %0\n\t"
               "mov %%eax, %1\n\t"
               "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "%rax", "%rbx", "%rcx", "%rdx"); 
asm   volatile ("CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "%rax", "%rbx", "%rcx", "%rdx"); 
asm   volatile("RDTSCP\n\t"
               "mov %%edx, %0\n\t"
               "mov %%eax, %1\n\t"
               "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "%rax", "%rbx", "%rcx", "%rdx");
#endif

    for (int i = 0; i < iterations; ++i) {
        generate_random_data(data1, range, n_values);
        generate_random_data(data2, range, n_values);

        clockdef t1 = std::chrono::high_resolution_clock::now();

#ifdef __linux__ 
    // unsigned long flags;
    // preempt_disable(); /*we disable preemption on our CPU*/
    // raw_local_irq_save(flags); /*we disable hard interrupts on our CPU*/  
    /*at this stage we exclusively own the CPU*/ 
#endif

#ifndef _MSC_VER 
    asm   volatile ("CPUID\n\t"
                    "RDTSC\n\t"
                    "mov %%edx, %0\n\t"
                    "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "%rax", "%rbx", "%rcx", "%rdx");
#endif
    // Call argument subroutine pointer.
    uint64_t ret = (*f)(data1, data2, n_bitmaps);

#ifndef _MSC_VER 
    asm   volatile("RDTSCP\n\t"
                   "mov %%edx, %0\n\t"
                   "mov %%eax, %1\n\t"
                   "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "%rax", "%rbx", "%rcx", "%rdx");
#endif
#ifdef __linux__ 
        // raw_local_irq_restore(flags);/*we enable hard interrupts on our CPU*/
        // preempt_enable();/*we enable preemption*/
#endif

        clockdef t2 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);

        // assert_truth(counters, flags_truth);
        // std::cerr << cycles_low <<"-" << cycles_high << " and " << cycles_low1 << "-" << cycles_high1 << std::endl;
        // std::cerr << "diff=" << (cycles_low1-cycles_low) << "->" << (cycles_low1-cycles_low)/(double)n << std::endl;
        uint64_t start = ( ((uint64_t)cycles_high  << 32) | cycles_low  );
        uint64_t end   = ( ((uint64_t)cycles_high1 << 32) | cycles_low1 );

        clocks.push_back(end - start);
        times.push_back(time_span.count());
    }

    uint64_t tot_cycles = 0, tot_time = 0;
    uint64_t min_c = std::numeric_limits<uint64_t>::max(), max_c = 0;
    for (int i = 0; i < clocks.size(); ++i) {
        tot_cycles += clocks[i];
        tot_time += times[i];
        min_c = std::min(min_c, clocks[i]);
        max_c = std::max(max_c, clocks[i]);
    }
    double mean_cycles = tot_cycles / (double)clocks.size();
    uint32_t mean_time = tot_time / (double)clocks.size();

    double variance = 0, stdDeviation = 0, mad = 0;
    for(int i = 0; i < clocks.size(); ++i) {
        variance += pow(clocks[i] - mean_cycles, 2);
        mad += std::abs(clocks[i] - mean_cycles);
    }
    mad /= clocks.size();
    variance /= clocks.size();
    stdDeviation = sqrt(variance);

    std::cout << name << "\t" << n_bitmaps << "\t" << 
        mean_cycles << "\t" <<
        min_c << "(" << min_c/mean_cycles << ")" << "\t" << 
        max_c << "(" << max_c/mean_cycles << ")" << "\t" <<
        stdDeviation << "\t" << 
        mad << "\t" << 
        mean_time << "\t" << 
        mean_cycles / n_bitmaps << "\t" << 
        ((n_bitmaps*2*sizeof(uint64_t)) / (1024*1024.0)) / (mean_time / 1000000000.0) << std::endl;

    // End timer and update times.
    //uint64_t cpu_cycles_after = get_cpu_cycles();
    
    unit.times += mean_time;
    unit.times_local = mean_time;
    //unit.cycles += (cpu_cycles_after - cpu_cycles_before);
    //unit.cycles_local = (cpu_cycles_after - cpu_cycles_before);
    unit.cycles += mean_cycles;
    unit.cycles_local = mean_cycles;
    unit.valid = 1;

    //std::cerr << cycles_low <<"-" << cycles_high << " and " << cycles_low1 << "-" << cycles_high1 << std::endl;
    //std::cerr << "diff=" << (cycles_low1-cycles_low) << "->" << (cycles_low1-cycles_low)/(double)n << std::endl;
    return 0;
}

int benchmark(uint32_t range, uint32_t n_values) {
    std::cout << "Range = [0," << range << ") with bits=" << n_values << std::endl;

    // Align some bitmaps.
    uint32_t n_bitmaps = ceil(range / 64.0);
    uint64_t* bitmaps  = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), n_bitmaps*sizeof(uint64_t));
    uint64_t* bitmaps2 = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), n_bitmaps*sizeof(uint64_t));
    
    generate_random_data(bitmaps, range, n_values);
    generate_random_data(bitmaps2, range, n_values);

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

    {
        // Align some bitmaps.
        uint32_t n_bitmaps = ceil(range / 64.0);
        uint64_t* bitmaps  = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), 65536*sizeof(uint64_t));
        uint64_t* bitmaps2 = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), 65536*sizeof(uint64_t));

        std::vector<uint32_t> ranges = {128,256,512,1024,2048,4096,8192,65536};

        for (int i = 0; i < ranges.size(); ++i) {
            bench_unit unit_intsec, unit_union, unit_diff;
            set_algebra_wrapper("intersect",STORM_get_intersect_count_func(n_bitmaps), 1000, bitmaps, bitmaps2, range, n_values, ranges[i], unit_intsec);
            set_algebra_wrapper("union",STORM_get_union_count_func(n_bitmaps), 1000, bitmaps, bitmaps2, range, n_values, ranges[i], unit_union);
            set_algebra_wrapper("diff",STORM_get_diff_count_func(n_bitmaps), 1000, bitmaps, bitmaps2, range, n_values, ranges[i], unit_diff);
        }

        // Clean up.
        STORM_aligned_free(bitmaps);
        STORM_aligned_free(bitmaps2);
    }

    // Clean up.
    STORM_aligned_free(bitmaps);
    STORM_aligned_free(bitmaps2);
    return 1;
}

int main(int argc, char **argv) { 
    benchmark(2048000, 50000);

    return EXIT_SUCCESS;
}