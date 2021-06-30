#include <new>

#include "ppl/kernel/x86/fp32/fc.h"
#include "ppl/kernel/x86/fp32/fc/fma/fc_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

fc_fp32_algo_info fc_algo_selector::select_algo(const ppl::common::dataformat_t &src_format, const fc_fp32_param &param, const ppl::common::isa_t &isa_flags)
{
    (void)src_format;

    static fc_fp32_algo_info unknown_info = {
        .algo_type = fc_fp32_algo::unknown,
        .isa       = ppl::common::ISA_undef};

    if (isa_flags & ppl::common::ISA_X86_FMA) {
        return (fc_fp32_algo_info){
            .algo_type = fc_fp32_algo::standard,
            .isa       = ppl::common::ISA_X86_FMA};
    } else {
        return unknown_info;
    }
}

fc_fp32_manager *fc_algo_selector::gen_algo(const fc_fp32_param &param, const fc_fp32_algo_info &algo_info, ppl::common::Allocator *allocator)
{
    fc_fp32_manager *fc_mgr = nullptr;
    if (algo_info.algo_type == fc_fp32_algo::standard &&
        algo_info.isa == ppl::common::ISA_X86_FMA) {
        fc_mgr = new fc_fp32_fma_manager(param, allocator);
    }

    return fc_mgr;
}

}}}; // namespace ppl::kernel::x86
