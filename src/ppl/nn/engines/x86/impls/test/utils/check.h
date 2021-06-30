#ifndef __ST_FP_H_
#define __ST_FP_H_

#include <stdint.h>

template <typename T>
bool check_array_error(T* input, T* ref, uint64_t len, T eps)
{
    for (uint64_t i = 0; i < len; ++i) {
        double err = double(input[i] - ref[i]);
        if (std::abs(err / ref[i]) > eps && std::abs(err) > eps) {
            std::cerr << "error[" << i << "]=" << input[i] << " ref:" << ref[i];
            return false;
        }
    }
    std::cerr << "pass";
    return true;
}

#endif