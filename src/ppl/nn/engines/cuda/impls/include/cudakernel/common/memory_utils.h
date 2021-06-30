#ifndef PPLCUDA_KERNEL_INCLUDE_MEMORY_UTILS_H_
#define PPLCUDA_KERNEL_INCLUDE_MEMORY_UTILS_H_
#define MAX_DIMENSION 7 // should be acquired from ppl.common
#include <vector>
#include <stdint.h>
#include <assert.h>
/*
  GArray: an small array for transferring parameters to GPU(device)
*/
template <typename T, int32_t capacity = MAX_DIMENSION>
struct GArray {
    GArray()
        : size_(0)
        , data_()
    {
    }

    GArray(int32_t size)
        : size_(size)
        , data_()
    {
        assert(size >= 0 && size <= capacity);
    }

    GArray(const std::vector<T>& vec)
        : GArray(static_cast<int32_t>(vec.size()))
    {
// std::is_trivially_copyable is not implemented in older versions of GCC
#if !defined(__GNUC__) || __GNUC__ >= 5
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
#endif
        memcpy(data_, vec.data(), vec.size() * sizeof(T));
    }

    void SetSize(int32_t size)
    {
        assert(size >= 0 && size <= capacity);
        size_ = size;
    }

    __host__ __device__ int32_t Size() const
    {
        return size_;
    }

    __host__ __device__ T& operator[](int32_t index)
    {
        return data_[index];
    }

    __host__ __device__ __forceinline__ const T& operator[](int32_t index) const
    {
        return data_[index];
    }

    __host__ __device__ T* Data()
    {
        return data_;
    }

    __host__ __device__ const T* Data() const
    {
        return data_;
    }

    static constexpr int32_t Capacity()
    {
        return capacity;
    };

private:
    int32_t size_;
    T data_[capacity];
};
#endif // PPLCUDA_KERNEL_INCLUDE_MEMORY_UTILS_H_