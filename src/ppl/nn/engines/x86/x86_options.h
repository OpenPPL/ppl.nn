#ifndef _ST_HPC_PPL_NN_ENGINES_X86_X86_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_X86_OPTIONS_H_

namespace ppl { namespace nn { namespace x86 {

enum {
    /**
       @brief disable avx512 support

       @note example:
       @code{.cpp}
       x86_engine->Configure(X86_CONF_DISABLE_AVX512);
       @endcode
    */
    X86_CONF_DISABLE_AVX512 = 0,

    /** max value */
    X86_CONF_MAX,
};

}}} // namespace ppl::nn::x86

#endif
