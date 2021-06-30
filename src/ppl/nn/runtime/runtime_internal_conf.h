#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_INTERNAL_CONF_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_INTERNAL_CONF_H_

namespace ppl { namespace nn {

struct RuntimeInternalConf {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    bool profiling_flag = false;
#endif
};

}} // namespace ppl::nn

#endif
