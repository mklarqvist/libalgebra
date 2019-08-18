#include "libalgebra.h"

#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include <vector>
#if !defined(_MSC_VER)
#include "getopt.h"
#endif

void generate_random_data(uint64_t* data, uint32_t range, uint32_t n) {
    // Clear data
    uint32_t n_bitmaps = ceil(n / 64.0);
    memset(data, 0, sizeof(uint64_t)*n_bitmaps);

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

int linux_set_algebra_wrapper(std::string name,
    STORM_compute_func f, 
    int iterations,
    uint64_t* STORM_RESTRICT data1, 
    uint64_t* STORM_RESTRICT data2, 
    uint32_t range,
    uint32_t n_values,
    uint32_t n_bitmaps, 
    bool verbose) 
{
    std::vector<int> evts;
    evts.push_back(PERF_COUNT_HW_CPU_CYCLES);
    evts.push_back(PERF_COUNT_HW_INSTRUCTIONS);
    evts.push_back(PERF_COUNT_HW_BRANCH_MISSES);
    evts.push_back(PERF_COUNT_HW_CACHE_REFERENCES);
    evts.push_back(PERF_COUNT_HW_CACHE_MISSES);
    evts.push_back(PERF_COUNT_HW_REF_CPU_CYCLES);
    LinuxEvents<PERF_TYPE_HARDWARE> unified(evts);
    std::vector<unsigned long long> results; // tmp buffer
    std::vector< std::vector<unsigned long long> > allresults;
    results.resize(evts.size());
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 0xFFFF);

    volatile uint64_t total = 0; // voltatile to prevent compiler to remove work through optimization    
    for (uint32_t i = 0; i < iterations; i++) {
        generate_random_data(data1, range, n_values);
        generate_random_data(data2, range, n_values);
        
        unified.start();
        // Call argument subroutine pointer.
        total += (*f)(data1, data2, n_bitmaps);
        unified.end(results);
        allresults.push_back(results);
    }

    std::vector<unsigned long long> mins = compute_mins(allresults);
    std::vector<double> avg = compute_averages(allresults);
    
    if (verbose) {
        printf("%s\t%u\t%.2f\t%.3f\t%.3f\t%llu\t%llu\t%llu\t%llu\t%llu\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n",
             name.c_str(),
             n_bitmaps,
             double(mins[1]) / mins[0], 
             double(mins[0]) / (2*n_bitmaps), 
             double(mins[1]) / (2*n_bitmaps),
             mins[0], 
             mins[1], 
             mins[2], 
             mins[3], 
             mins[4],
             avg[0], 
             avg[1], 
             avg[2], 
             avg[3], 
             avg[4]);
    } else {
        printf("%s-%u:\n",name.c_str(),n_bitmaps);
        printf("instructions per cycle %4.2f, cycles per 64-bit word:  %4.3f, "
               "instructions per 64-bit word %4.3f \n",
                double(mins[1]) / mins[0], double(mins[0]) / (2*n_bitmaps), double(mins[1]) / (2*n_bitmaps));
        // first we display mins
        printf("min: %8llu cycles, %8llu instructions, \t%8llu branch mis., %8llu "
               "cache ref., %8llu cache mis.\n",
                mins[0], mins[1], mins[2], mins[3], mins[4]);
        printf("avg: %8.1f cycles, %8.1f instructions, \t%8.1f branch mis., %8.1f "
               "cache ref., %8.1f cache mis.\n",
                avg[0], avg[1], avg[2], avg[3], avg[4]);
    }

    return 1;
}

int linux_popcount_wrapper(std::string name,
    STORM_popcnt_func f, 
    int iterations,
    uint64_t* data,
    uint32_t range,
    uint32_t n_values,
    uint32_t n_bitmaps,
    bool verbose) 
{
    std::vector<int> evts;
    evts.push_back(PERF_COUNT_HW_CPU_CYCLES);
    evts.push_back(PERF_COUNT_HW_INSTRUCTIONS);
    evts.push_back(PERF_COUNT_HW_BRANCH_MISSES);
    evts.push_back(PERF_COUNT_HW_CACHE_REFERENCES);
    evts.push_back(PERF_COUNT_HW_CACHE_MISSES);
    evts.push_back(PERF_COUNT_HW_REF_CPU_CYCLES);
    LinuxEvents<PERF_TYPE_HARDWARE> unified(evts);
    std::vector<unsigned long long> results; // tmp buffer
    std::vector< std::vector<unsigned long long> > allresults;
    results.resize(evts.size());
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 0xFFFF);

    volatile uint64_t total = 0; // voltatile to prevent compiler to remove work through optimization    
    for (uint32_t i = 0; i < iterations; i++) {
        generate_random_data(data, range, n_values);
        size_t n_vals = n_bitmaps*8;
        
        unified.start();
        // Call argument subroutine pointer.
        total += (*f)((uint8_t*)data, n_vals);
        unified.end(results);
        allresults.push_back(results);
    }

    std::vector<unsigned long long> mins = compute_mins(allresults);
    std::vector<double> avg = compute_averages(allresults);
    
    if (verbose) {
        printf("%s\t%u\t%.2f\t%.3f\t%.3f\t%llu\t%llu\t%llu\t%llu\t%llu\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n",
             name.c_str(),
             n_bitmaps,
             double(mins[1]) / mins[0], 
             double(mins[0]) / n_bitmaps, 
             double(mins[1]) / n_bitmaps,
             mins[0], 
             mins[1], 
             mins[2], 
             mins[3], 
             mins[4],
             avg[0], 
             avg[1], 
             avg[2], 
             avg[3], 
             avg[4]);
    } else {
        printf("%s-%u:\n",name.c_str(),n_bitmaps);
        printf("instructions per cycle %4.2f, cycles per 64-bit word:  %4.3f, "
               "instructions per 64-bit word %4.3f \n",
                double(mins[1]) / mins[0], double(mins[0]) / n_bitmaps, double(mins[1]) / n_bitmaps);
        // first we display mins
        printf("min: %8llu cycles, %8llu instructions, \t%8llu branch mis., %8llu "
               "cache ref., %8llu cache mis.\n",
                mins[0], mins[1], mins[2], mins[3], mins[4]);
        printf("avg: %8.1f cycles, %8.1f instructions, \t%8.1f branch mis., %8.1f "
               "cache ref., %8.1f cache mis.\n",
                avg[0], avg[1], avg[2], avg[3], avg[4]);
    }

    return 1;
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

#if !defined(__clang__) && !defined(_MSC_VER)
__attribute__((optimize("no-tree-vectorize")))
#endif
uint64_t intersect_scalar_naive_nosimd(const uint64_t* STORM_RESTRICT data1,const uint64_t* STORM_RESTRICT data2, size_t len) {
    uint64_t total = 0;
    // for (int i = 0; i < len; ++i) {
    //     total += STORM_popcount64(data1[i] & data2[i]);
    // }

    uint64_t diff;
    uint8_t* b8 = (uint8_t*)&diff;
    for (int i = 0; i < len; ++i) {
        // total += STORM_popcount64(data1[i] & data2[i]);
        diff = data1[i] & data2[i];
        total += STORM_popcnt_lookup8bit[b8[0]];
        total += STORM_popcnt_lookup8bit[b8[1]];
        total += STORM_popcnt_lookup8bit[b8[2]];
        total += STORM_popcnt_lookup8bit[b8[3]];
        total += STORM_popcnt_lookup8bit[b8[4]];
        total += STORM_popcnt_lookup8bit[b8[5]];
        total += STORM_popcnt_lookup8bit[b8[6]];
        total += STORM_popcnt_lookup8bit[b8[7]];
    }

    return total;
}

#if !defined(__clang__) && !defined(_MSC_VER)
__attribute__((optimize("no-tree-vectorize")))
#endif
uint64_t union_scalar_naive_nosimd(const uint64_t* STORM_RESTRICT data1,const uint64_t* STORM_RESTRICT data2, size_t len) {
    uint64_t total = 0;
    // for (int i = 0; i < len; ++i) {
    //     total += STORM_popcount64(data1[i] | data2[i]);
    // }

    uint64_t diff;
    uint8_t* b8 = (uint8_t*)&diff;
    for (int i = 0; i < len; ++i) {
        // total += STORM_popcount64(data1[i] | data2[i]);
        diff = data1[i] | data2[i];
        total += STORM_popcnt_lookup8bit[b8[0]];
        total += STORM_popcnt_lookup8bit[b8[1]];
        total += STORM_popcnt_lookup8bit[b8[2]];
        total += STORM_popcnt_lookup8bit[b8[3]];
        total += STORM_popcnt_lookup8bit[b8[4]];
        total += STORM_popcnt_lookup8bit[b8[5]];
        total += STORM_popcnt_lookup8bit[b8[6]];
        total += STORM_popcnt_lookup8bit[b8[7]];
    }

    return total;
}

#if !defined(__clang__) && !defined(_MSC_VER)
__attribute__((optimize("no-tree-vectorize")))
#endif
uint64_t diff_scalar_naive_nosimd(const uint64_t* STORM_RESTRICT data1,const uint64_t* STORM_RESTRICT data2, size_t len) {
    uint64_t total = 0;
    // for (int i = 0; i < len; ++i) {
    //     total += STORM_popcount64(data1[i] ^ data2[i]);
    // }

    uint64_t diff;
    uint8_t* b8 = (uint8_t*)&diff;
    for (int i = 0; i < len; ++i) {
        // total += STORM_popcount64(data1[i] ^ data2[i]);
        diff = data1[i] ^ data2[i];
        total += STORM_popcnt_lookup8bit[b8[0]];
        total += STORM_popcnt_lookup8bit[b8[1]];
        total += STORM_popcnt_lookup8bit[b8[2]];
        total += STORM_popcnt_lookup8bit[b8[3]];
        total += STORM_popcnt_lookup8bit[b8[4]];
        total += STORM_popcnt_lookup8bit[b8[5]];
        total += STORM_popcnt_lookup8bit[b8[6]];
        total += STORM_popcnt_lookup8bit[b8[7]];
    }

    return total;
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
    size_t n_bitmaps, 
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
    generate_random_data(data1, range, n_values);
    generate_random_data(data2, range, n_values);

    volatile uint64_t total = 0; // voltatile to prevent compiler to remove work through optimization
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

    for (int i = 0; i < iterations; ++i) {
        // Call argument subroutine pointer.
        total += (*f)(data1, data2, n_bitmaps);
    }

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

    uint64_t start = ( ((uint64_t)cycles_high  << 32) | cycles_low  );
    uint64_t end   = ( ((uint64_t)cycles_high1 << 32) | cycles_low1 );

    double mean_cycles = (end - start) / (double)iterations;
    uint32_t mean_time = time_span.count() / (double)iterations;

    std::cout << name << "\t" << n_bitmaps << "\t" << total << "\t" << 
        mean_cycles << "\t" <<
        mean_time << "\t" << 
        mean_cycles / n_bitmaps << "\t" << 
        ((n_bitmaps*2*sizeof(uint64_t)) / (1024*1024.0)) / (mean_time / 1000000000.0) << std::endl;
    
    unit.times += mean_time;
    unit.times_local = mean_time;
    unit.cycles += mean_cycles;
    unit.cycles_local = mean_cycles;
    unit.valid = 1;

    return 0;
}

int popcount_wrapper(std::string name,
    STORM_popcnt_func f, 
    int iterations,
    uint64_t* data,
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
    generate_random_data(data, range, n_values);

    volatile uint64_t total = 0; // voltatile to prevent compiler to remove work through optimization
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

    size_t n_b = n_bitmaps*8;
    for (int i = 0; i < iterations; ++i) {
        // Call argument subroutine pointer.
        total += (*f)((uint8_t*)data, n_b);
    }

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

    uint64_t start = ( ((uint64_t)cycles_high  << 32) | cycles_low  );
    uint64_t end   = ( ((uint64_t)cycles_high1 << 32) | cycles_low1 );

    double mean_cycles = (end - start) / (double)iterations;
    uint32_t mean_time = time_span.count() / (double)iterations;

    std::cout << name << "\t" << n_bitmaps << "\t" << total << "\t" << 
        mean_cycles << "\t" <<
        mean_time << "\t" << 
        mean_cycles / n_bitmaps << "\t" << 
        ((n_bitmaps*2*sizeof(uint64_t)) / (1024*1024.0)) / (mean_time / 1000000000.0) << std::endl;
    
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

    std::vector<uint32_t> ranges = {4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576};
    std::vector<uint32_t> reps;
    if (n_repetitions <= 0) {
        reps = {5000,5000,5000,5000,5000,2500,2500,2500,2500,2500,50,50,50,50,50,50,50,25,25,25};
    } else {
        reps = std::vector<uint32_t>(ranges.size(), n_repetitions);
    }

    if (use_perf) {
#ifndef __linux__ 
        std::cerr << "perf counter are only available on Linux systems!" << std::endl;
        exit(EXIT_FAILURE);
#endif
        printf("Algorithm\tWords\tInstructions/cycle\tCycles/word\tInstructions/word\tMinCycles\tMinInstructions\tMinBranchMiss\tMinCacheRef\tminCacheMiss\tAvgCycles\tAvgInstructions\tAvgBranchMiss\tAvgCacheRef\tAvgCacheMiss\n");
    }


    for (int i = 0; i < ranges.size(); ++i) {
        bench_unit unit_intsec, unit_union, unit_diff;
        
        if (use_perf) {
#ifdef __linux__ 
            linux_popcount_wrapper("popcount-naive",&popcount_scalar_naive_nosimd, reps[i], bitmaps, ranges[i], ranges[i], ranges[i], true);
            linux_popcount_wrapper("popcount",&STORM_popcnt, reps[i], bitmaps, ranges[i], ranges[i], ranges[i], true);
            linux_set_algebra_wrapper("intersect-naive",&intersect_scalar_naive_nosimd, reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], true);
            linux_set_algebra_wrapper("intersect",STORM_get_intersect_count_func(ranges[i]), reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], true);
            linux_set_algebra_wrapper("union-naive",&union_scalar_naive_nosimd, reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], true);
            linux_set_algebra_wrapper("union",STORM_get_union_count_func(ranges[i]), reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], true);
            linux_set_algebra_wrapper("diff-naive",&diff_scalar_naive_nosimd, reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], true);
            linux_set_algebra_wrapper("diff",STORM_get_diff_count_func(ranges[i]), reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], true);
#else
            std::cerr << "perf counter are only available on Linux systems!" << std::endl;
            exit(EXIT_FAILURE);
#endif
        } else {
            popcount_wrapper("popcount-naive",&popcount_scalar_naive_nosimd, reps[i], bitmaps, ranges[i], ranges[i], ranges[i], unit_intsec);
            popcount_wrapper("popcount",&STORM_popcnt, reps[i], bitmaps, ranges[i], ranges[i], ranges[i], unit_intsec);
            set_algebra_wrapper("intersect-naive",&intersect_scalar_naive_nosimd, reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], unit_intsec);
            set_algebra_wrapper("intersect",STORM_get_intersect_count_func(ranges[i]), reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], unit_intsec);
            set_algebra_wrapper("union-naive",&union_scalar_naive_nosimd, reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], unit_intsec);
            set_algebra_wrapper("union",STORM_get_union_count_func(ranges[i]), reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], unit_union);
            set_algebra_wrapper("diff-naive",&diff_scalar_naive_nosimd, reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], unit_intsec);
            set_algebra_wrapper("diff",STORM_get_diff_count_func(ranges[i]), reps[i], bitmaps, bitmaps2, ranges[i], ranges[i], ranges[i], unit_diff);
        }
    }

    // Clean up.
    STORM_aligned_free(bitmaps);
    STORM_aligned_free(bitmaps2);
    
    return 1;
}

int main(int argc, char **argv) {
#if !defined(_MSC_VER)
    bool verbose = false;
    bool perf_subsystem = false;
    int c;
    int n_repetitions = -1;

    while ((c = getopt(argc, argv, "vpr:")) != -1) {
        switch (c) {
        case 'r':
            n_repetitions = atoi(optarg);
            break;
        case 'v':
            verbose = true;
            break;
        case 'p':
            perf_subsystem = true;
            break;
        default:
            abort();
        }
    }

    benchmark(n_repetitions, perf_subsystem);
#else
    int n_repetitions = -1;
    if (argc > 2) {
        n_repetitions = std::atoi(argv[1]);
    }
    benchmark(n_repetitions, false);
#endif

    return EXIT_SUCCESS;
}