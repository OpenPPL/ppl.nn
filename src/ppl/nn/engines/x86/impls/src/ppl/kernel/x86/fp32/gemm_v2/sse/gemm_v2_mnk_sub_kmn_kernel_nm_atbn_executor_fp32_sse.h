#ifndef __ST_PPL_KERNEL_X86_FP32_GEMM_V2_GEMM_V2_MNK_SUB_KMN_KERNEL_NM_ATBN_EXECUTOR_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_GEMM_V2_GEMM_V2_MNK_SUB_KMN_KERNEL_NM_ATBN_EXECUTOR_FP32_SSE_H_

#include <string.h> // for memcpy

#include "ppl/kernel/x86/fp32/gemm_v2.h"
#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

class gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse : public gemm_v2_executor_fp32 {
public:
    gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse()
    {
        // default internal param
        blk_partition_.m_blk_len = 128; // 256 KB used in L2/L3
        blk_partition_.n_blk_len = 192;
        blk_partition_.k_blk_len = 128;

        blk_partition_.m_sub_blk_len = 32; // 26 KB used in L1
        blk_partition_.n_sub_blk_len = 48;
        blk_partition_.k_sub_blk_len = 64;

        blk_partition_.m_kernel_blk_len = 4; // only support 4x12 kernel now
        blk_partition_.n_kernel_blk_len = 12;
    }
    virtual ~gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse() {}

    void set_internal_param(const std::vector<uint8_t>& internal_param) override final
    {
        memcpy(&blk_partition_, internal_param.data(), min(internal_param.size(), sizeof(blk_partition)));
    }

    const void* get_internal_param_ptr(void) override final
    {
        return (const void*)&blk_partition_;
    }

    virtual uint64_t get_internal_param_bytes(void) override final
    {
        return sizeof(blk_partition);
    }

    uint64_t get_buffer_bytes(void) const override final
    {
        return get_buffer_len_per_thread() * PPL_OMP_MAX_THREADS() * sizeof(float);
    }

    common::RetCode optimize(void) override final
    {
        return common::RC_SUCCESS;
    } // TODO: add optimize

    common::RetCode execute(void) override final;

private:
    // buffer related functions
    inline uint64_t get_a_buffer_len(void) const
    {
        return blk_partition_.m_blk_len * blk_partition_.k_blk_len;
    }
    inline uint64_t get_b_buffer_len(void) const
    {
        return blk_partition_.k_blk_len * blk_partition_.n_blk_len;
    }
    inline uint64_t get_dst_buffer_len(void) const
    {
        return blk_partition_.m_blk_len * blk_partition_.n_blk_len;
    }
    inline uint64_t get_buffer_len_per_thread(void) const
    {
        return get_a_buffer_len() + get_b_buffer_len() + get_dst_buffer_len() + 16;
    }

    // execute related functions
    inline void load_a_data(const float* src, const int32_t m_len, const int32_t k_len, float* dst);
    inline void load_b_data(const float* src, const int32_t n_len, const int32_t k_len, float* dst);
    inline void store_dst_data(const float* src, const int32_t m_len, const int32_t n_len, const float* C, float* dst);
    inline void execute_sub_blk(const float* A, const float* B, const int32_t m_len, const int32_t n_len, const int32_t k_len, float* dst);

private:
    struct blk_partition {
        // L2/L3 blk
        int32_t m_blk_len;
        int32_t n_blk_len;
        int32_t k_blk_len;
        // L1 blk
        int32_t m_sub_blk_len;
        int32_t n_sub_blk_len;
        int32_t k_sub_blk_len;
        // register blk
        int32_t m_kernel_blk_len;
        int32_t n_kernel_blk_len;
    };

    blk_partition blk_partition_;
};

}}} // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_FP32_GEMM_V2_GEMM_V2_MNK_SUB_KMN_KERNEL_NM_ATBN_EXECUTOR_FP32_SSE_H_