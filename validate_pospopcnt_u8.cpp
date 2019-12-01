#include "libalgebra.h"

#include <iostream>
#include <random>
#include <vector>
#include <cstdint>
#include <cassert>


class Application {

    std::vector<uint8_t> input;
    std::vector<uint32_t> reference;
    std::ostream& out;

public:
    Application() : out(std::cout) {}

    bool run() {
        initalize_input();

        reference = pospopcnt_u8(STORM_pospopcnt_u8_scalar_naive);

        bool ok = true;
        ok &= validate("umul128",              &STORM_pospopcnt_u8_scalar_umul128_unroll2);
        ok &= validate("SSE SAD",              &STORM_pospopcnt_u8_sse_sad);
        ok &= validate("SSE blend unroll8",    &STORM_pospopcnt_u8_sse_blend_popcnt_unroll8);
        ok &= validate("SSE Harley-Seal",      &STORM_pospopcnt_u8_sse_harley_seal);
    
        return ok;
    }
 
private:
    void initalize_input() {
        std::uniform_int_distribution<uint8_t> distr(0, 255);
        std::random_device rd;
        std::mt19937 eng(rd()); // seed the generator
        eng.seed(42);

        input.resize(1024);
        for (auto& word: input)
            word = distr(eng);
    }

    bool validate(const std::string& name, STORM_pospopcnt_u8_func f) {
        out << "Checking " << name << " " << std::flush;
        std::vector<uint32_t> histogram{pospopcnt_u8(f)};
        if (compare(histogram)) {
            out << "OK\n";
            return true;
        } else {
            out << '\n';
            dump(reference, out) << '\n';
            dump(histogram, out) << '\n';
            return false;
        }
    }

    std::vector<uint32_t> pospopcnt_u8(STORM_pospopcnt_u8_func f) {
        std::vector<uint32_t> histogram;
        histogram.resize(8);
        f(input.data(), input.size(), histogram.data());

        return histogram;
    }

    bool compare(const std::vector<uint32_t>& histogram) {
        assert(histogram.size() == 8);
        bool valid = true;
        for (size_t i=0; i < 8; i++) {
            const uint32_t val = histogram[i];
            const uint32_t ref = reference[i];
            if (val != ref) {
                //out << "difference at #" << i << " result=" << val << " reference=" << ref << '\n';
                valid = false;
            }
        }

        return valid;
    }

    std::ostream& dump(const std::vector<uint32_t>& histogram, std::ostream& os) {
        os << '[';
        if (!histogram.empty()) {
            os << histogram[0];
            for (size_t i=1; i < histogram.size(); i++)
                os << ", " << histogram[i];
        }
        os << ']';

        return os;
    }
};


int main() {
    Application app;
    return app.run() ? EXIT_SUCCESS : EXIT_FAILURE;
}
