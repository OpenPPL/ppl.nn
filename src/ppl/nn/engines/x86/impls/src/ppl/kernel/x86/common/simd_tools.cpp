#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

void set_denormals_zero(const int32_t on) {
    if (ppl::common::GetCpuISA() & ppl::common::ISA_X86_SSE) {
        PRAGMA_OMP_PARALLEL()
        {
            if (on) {
                _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
                _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
            } else {
                _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
                _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
            }
        }
    }
}

}}};
