#ifndef __ST_PPL_KERNEL_X86_COMMON_MATH_H_
#define __ST_PPL_KERNEL_X86_COMMON_MATH_H_

#include <algorithm>
#include <type_traits>
#include <stdint.h>

namespace ppl { namespace kernel { namespace x86 {

template <typename T>
inline const T &max(const T &a, const T &b)
{
    return a > b ? a : b;
}

template <typename T>
inline const T &min(const T &a, const T &b)
{
    return a < b ? a : b;
}

template <typename T>
inline const T abs(const T &x)
{
    return x > 0 ? x : (-x);
}

template <typename T0, typename T1>
inline T0 div_up(const T0 &a, const T1 &b)
{
    const T0 tb = static_cast<T0>(b);
    return (a + tb - static_cast<T0>(1)) / tb;
}

template <typename T0, typename T1>
inline T0 round(const T0 &a, const T1 &b)
{
    const T0 tb = static_cast<T0>(b);
    return a / tb * tb;
}

template <typename T0, typename T1>
inline T0 round_up(const T0 &a, const T1 &b)
{
    const T0 tb = static_cast<T0>(b);
    return (a + tb - static_cast<T0>(1)) / tb * tb;
}

template <typename T0, typename T1>
inline T0 mod_up(const T0 &a, const T1 &m)
{
    static_assert(
        false || std::is_same<T0, int64_t>::value || std::is_same<T0, int32_t>::value,
        "only allow int64_t/int32_t");
    const T0 tm = static_cast<T0>(m);
    return (a % tm == 0 && a != 0) ? tm : a % tm;
}

template <typename DataType, typename IndexType>
inline void argsort(const DataType *src, IndexType *indices, const int64_t length, const bool dec = true)
{
    for (int64_t i = 0; i < length; ++i)
        indices[i] = static_cast<IndexType>(i);
    if (dec) {
        std::stable_sort(indices, indices + length,
            [&src](const IndexType &ind0, const IndexType &ind1) { return src[ind0] > src[ind1]; });
    } else {
        std::stable_sort(indices, indices + length,
            [&src](const IndexType &ind0, const IndexType &ind1) { return src[ind0] < src[ind1]; });
    }
}

}}}; // namespace ppl::kernel::x86

#endif
