#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_GEMM_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_GEMM_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct GemmParam {
    int32_t num_output;
    int32_t bias_term; // 0 or 1

    // version onnx
    float alpha;
    float beta;
    int32_t transA;
    int32_t transB;
    int32_t N; // for converted mat B

    bool operator==(const GemmParam& p) const {
        return this->num_output == p.num_output && this->bias_term == p.bias_term && this->alpha == p.alpha &&
            this->beta == p.beta && this->transA == p.transA && this->transB == p.transB && this->N == p.N;
    }
};

}}} // namespace ppl::nn::common

#endif
