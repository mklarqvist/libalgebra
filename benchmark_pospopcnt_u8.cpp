#include "libalgebra.h"

#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include <vector>
#if !defined(_MSC_VER)
#include "getopt.h"
#endif

uint64_t* generate_random_data(uint32_t n_bitmaps) {
    // Clear data
    // uint32_t n_bitmaps = ceil(n / 64.0);
    // memset(data, 0, sizeof(uint64_t)*n_bitmaps);
    uint64_t* mem = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), n_bitmaps*sizeof(uint64_t));

    // PRNG
    std::uniform_int_distribution<uint32_t> distr(0, std::numeric_limits<uint32_t>::max()-1); // right inclusive
    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

     // Generate some random data.
    uint32_t n_unique = 0;
    // while (n_unique < n) {
    for (int i = 0; i < n_bitmaps; ++i) {
        uint32_t val1 = distr(eng);
        uint32_t val2 = distr(eng);
        uint64_t x = ((uint64_t)val1 << 32) | val2;
        mem[i] = x;
    }

    return mem;
}

#if !defined(__clang__) && !defined(_MSC_VER)
__attribute__((optimize("no-tree-vectorize")))
#endif
uint64_t popcount_scalar_naive_nosimd(const uint8_t* data, size_t len) {
    uint64_t total = 0;
    // for (int i = 0; i < len; ++i) {
    //     total += STORM_popcount64(data1[i] & data2[i]);
    // }
    // assert(len % 8 == 0);

    for (int j = 0; j < len; j += 8) {
        // total += STORM_popcount64(data[i]);
        // diff = data1[i] & data2[i];
        total += STORM_popcnt_lookup8bit[data[j+0]];
        total += STORM_popcnt_lookup8bit[data[j+1]];
        total += STORM_popcnt_lookup8bit[data[j+2]];
        total += STORM_popcnt_lookup8bit[data[j+3]];
        total += STORM_popcnt_lookup8bit[data[j+4]];
        total += STORM_popcnt_lookup8bit[data[j+5]];
        total += STORM_popcnt_lookup8bit[data[j+6]];
        total += STORM_popcnt_lookup8bit[data[j+7]];
    }

    return total;
}

#ifdef __linux__

#include <asm/unistd.h>       // for __NR_perf_event_open
#include <linux/perf_event.h> // for perf event constants
#include <sys/ioctl.h>        // for ioctl
#include <unistd.h>           // for syscall
#include <iostream>
#include <cerrno>  // for errno
#include <cstring> // for memset
#include <stdexcept>

#include <vector>

template <int TYPE = PERF_TYPE_HARDWARE> 
class LinuxEvents {
    int fd;
    bool working;
    perf_event_attr attribs;
    int num_events;
    std::vector<uint64_t> temp_result_vec;
    std::vector<uint64_t> ids;

public:
    explicit LinuxEvents(std::vector<int> config_vec) : fd(0), working(true) {
        memset(&attribs, 0, sizeof(attribs));
        attribs.type = TYPE;
        attribs.size = sizeof(attribs);
        attribs.disabled = 1;
        attribs.exclude_kernel = 1;
        attribs.exclude_hv = 1;

        attribs.sample_period = 0;
        attribs.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
        const int pid = 0;  // the current process
        const int cpu = -1; // all CPUs
        const unsigned long flags = 0;

        int group = -1; // no group
        num_events = config_vec.size();
        uint32_t i = 0;
        for (auto config : config_vec) {
            attribs.config = config;
            fd = syscall(__NR_perf_event_open, &attribs, pid, cpu, group, flags);
            if (fd == -1) {
                report_error("perf_event_open");
            }
                ioctl(fd, PERF_EVENT_IOC_ID, &ids[i++]);
                if (group == -1) {
                group = fd;
            }
        }

        temp_result_vec.resize(num_events * 2 + 1);
    }

    ~LinuxEvents() { close(fd); }

    inline void start() {
        if (ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == -1) {
            report_error("ioctl(PERF_EVENT_IOC_RESET)");
        }

        if (ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1) {
            report_error("ioctl(PERF_EVENT_IOC_ENABLE)");
        }
    }

    inline void end(std::vector<unsigned long long> &results) {
        if (ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) {
            report_error("ioctl(PERF_EVENT_IOC_DISABLE)");
        }

        if (read(fd, &temp_result_vec[0], temp_result_vec.size() * 8) == -1) {
            report_error("read");
        }
        // our actual results are in slots 1,3,5, ... of this structure
        // we really should be checking our ids obtained earlier to be safe
        for (uint32_t i = 1; i < temp_result_vec.size(); i += 2) {
            results[i / 2] = temp_result_vec[i];
        }
    }

private:
    void report_error(const std::string &context) {
    if (working)
        std::cerr << (context + ": " + std::string(strerror(errno))) << std::endl;
        working = false;
    }
};

std::vector<unsigned long long>
compute_mins(std::vector< std::vector<unsigned long long> > allresults) {
    if (allresults.size() == 0)
        return std::vector<unsigned long long>();
    
    std::vector<unsigned long long> answer = allresults[0];
    
    for (size_t k = 1; k < allresults.size(); k++) {
        assert(allresults[k].size() == answer.size());
        for (size_t z = 0; z < answer.size(); z++) {
            if (allresults[k][z] < answer[z])
                answer[z] = allresults[k][z];
        }
    }
    return answer;
}

std::vector<double>
compute_averages(std::vector< std::vector<unsigned long long> > allresults) {
    if (allresults.size() == 0)
        return std::vector<double>();
    
    std::vector<double> answer(allresults[0].size());
    
    for (size_t k = 0; k < allresults.size(); k++) {
        assert(allresults[k].size() == answer.size());
        for (size_t z = 0; z < answer.size(); z++) {
            answer[z] += allresults[k][z];
        }
    }

    for (size_t z = 0; z < answer.size(); z++) {
        answer[z] /= allresults.size();
    }
    return answer;
}
#endif // end is linux

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

// Definition for microsecond timer.
typedef std::chrono::high_resolution_clock::time_point clockdef;

int pospopcount_u8_wrapper(std::string name,
    STORM_pospopcnt_u8_func f, 
    int iterations,
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
    uint64_t* mem = generate_random_data(n_values);

    volatile uint64_t total = 0; // voltatile to prevent compiler to remove work through optimization
    clockdef t1 = std::chrono::high_resolution_clock::now();

#ifndef _MSC_VER 
    asm   volatile ("CPUID\n\t"
                    "RDTSC\n\t"
                    "mov %%edx, %0\n\t"
                    "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "%rax", "%rbx", "%rcx", "%rdx");
#endif
    
    uint32_t histogram[16];
    for (int i = 0; i < 16; i++)
        histogram[i] = 0;

    for (int i = 0; i < iterations; ++i) {
        // Call argument subroutine pointer.
        f((uint8_t*)mem, n_values, histogram);
        for (int j=0; j < 16; j++)
            total += histogram[j];
    }

#ifndef _MSC_VER 
    asm   volatile("RDTSCP\n\t"
                   "mov %%edx, %0\n\t"
                   "mov %%eax, %1\n\t"
                   "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "%rax", "%rbx", "%rcx", "%rdx");
#endif

    const clockdef t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);

    STORM_aligned_free(mem);

    uint64_t start = ( ((uint64_t)cycles_high  << 32) | cycles_low  );
    uint64_t end   = ( ((uint64_t)cycles_high1 << 32) | cycles_low1 );

    double mean_cycles = (end - start) / (double)iterations;
    double mean_time = time_span.count() / (double)iterations;

    printf("%-20s: %5d  cycles = %10.2f  time = %10.2f\n",
           name.c_str(),
           n_values,
           mean_cycles,
           mean_time);
    
    unit.times += mean_time;
    unit.times_local = mean_time;
    unit.cycles += mean_cycles;
    unit.cycles_local = mean_cycles;
    unit.valid = 1;

    return 0;
}

int benchmark(int n_repetitions, bool use_perf = false) {
    // Align some bitmaps.
    uint64_t* bitmaps  = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), 1048576*sizeof(uint64_t));
    uint64_t* bitmaps2 = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), 1048576*sizeof(uint64_t));

    //std::vector<uint32_t> ranges = {4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576};
    std::vector<uint32_t> ranges = {256,1024,4096,8192};
    std::vector<uint32_t> reps;
    if (n_repetitions <= 0) {
        reps = {5000,5000,5000,5000,5000,2500,2500,2500,2500,2500,150,150,150,150,150,150,150,100,100,100};
    } else {
        reps = std::vector<uint32_t>(ranges.size(), n_repetitions);
    }

    for (int i = 0; i < ranges.size(); ++i) {
        bench_unit unit_intsec, unit_union, unit_diff;
        
        puts("");
        pospopcount_u8_wrapper("scalar",               &STORM_pospopcnt_u8_scalar_naive, reps[i], ranges[i], ranges[i], ranges[i], unit_intsec);
        pospopcount_u8_wrapper("umul128",              &STORM_pospopcnt_u8_scalar_umul128_unroll2, reps[i], ranges[i], ranges[i], ranges[i], unit_intsec);
        pospopcount_u8_wrapper("SSE SAD",              &STORM_pospopcnt_u8_sse_sad, reps[i], ranges[i], ranges[i], ranges[i], unit_intsec);
        pospopcount_u8_wrapper("SSE blend unroll8",    &STORM_pospopcnt_u8_sse_blend_popcnt_unroll8, reps[i], ranges[i], ranges[i], ranges[i], unit_intsec);
        pospopcount_u8_wrapper("SSE Harley-Seal",      &STORM_pospopcnt_u8_sse_harley_seal, reps[i], ranges[i], ranges[i], ranges[i], unit_intsec);
        pospopcount_u8_wrapper("SSE variant 1",        &STORM_pospopcnt_u8_sse_variant1, reps[i], ranges[i], ranges[i], ranges[i], unit_intsec);
    }

    // Clean up.
    STORM_aligned_free(bitmaps);
    STORM_aligned_free(bitmaps2);
    
    return 1;
}

int main(int argc, char **argv) {
    int n_repetitions = -1;
    if (argc > 2) {
        n_repetitions = std::atoi(argv[1]);
    }
    benchmark(n_repetitions);

    return EXIT_SUCCESS;
}
