#ifndef _ST_HPC_PPL_NN_PARAMS_PPL_SHAPE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PPL_SHAPE_PARAM_H_

#include "ppl/nn/common/logger.h"
#include "ppl/nn/common/types.h"

#include <stdint.h>

#include <vector>
#include <string>

namespace ppl { namespace nn { namespace common {

struct ShapeMatrix {
    static const int64_t MAXDIMSIZE = 8;

    ShapeMatrix() {
        for (uint32_t i = 0; i <= MAXDIMSIZE; ++i) {
            for (uint32_t j = 0; j <= MAXDIMSIZE; ++j) {
                numerator[i][j] = (i == j ? 1 : 0);
                denominator[i][j] = 1;
            }
        }
    }

    void Append(const ShapeMatrix& other) {
        if (real_dim < 0 || other.real_dim < 0) {
            LOG(ERROR) << "Matrix has unknown dim count and cannot append.";
            return;
        }
        for (uint32_t i = 0; i < other.real_dim; ++i) {
            for (uint32_t j = 0; j <= MAXDIMSIZE; ++j) {
                numerator[i + real_dim][j] = other.numerator[i][j];
                denominator[i + real_dim][j] = other.denominator[i][j];
            }
        }
        real_dim += other.real_dim;
    }

    void Append(int64_t value) {
        if (real_dim < 0) {
            LOG(ERROR) << "Matrix has unknown dim count and cannot append.";
            return;
        }
        for (uint32_t j = 0; j < MAXDIMSIZE; ++j) {
            numerator[real_dim][j] = 0;
            denominator[real_dim][j] = 1;
        }
        numerator[real_dim][MAXDIMSIZE] = value;
        denominator[real_dim][MAXDIMSIZE] = 1;
        real_dim += 1;
    }

    void Gather(int64_t begin, int64_t end) {
        real_dim = end - begin;
        for (uint32_t i = 0; i < real_dim; ++i) {
            for (uint32_t j = 0; j < MAXDIMSIZE; ++j) {
                numerator[i][j] = numerator[begin + i][j];
                denominator[i][j] = denominator[begin + i][j];
            }
        }
    }

    void Arithmetic(const ShapeMatrix& other, std::string name) {
        if (other.real_dim != real_dim) {
            LOG(ERROR) << "Matrix has incorrect dim count for arithmetic op.";
            return;
        }
        for (uint32_t i = 0; i < real_dim; ++i) {
            if (name == "Add") {
                numerator[i][MAXDIMSIZE] += other.numerator[i][MAXDIMSIZE] * denominator[i][MAXDIMSIZE];
            } else if (name == "Sub") {
                numerator[i][MAXDIMSIZE] -= other.numerator[i][MAXDIMSIZE] * denominator[i][MAXDIMSIZE];
            } else if (name == "Div") {
                for (uint32_t j = 0; j <= MAXDIMSIZE; ++j) {
                    numerator[i][j] *= other.denominator[i][MAXDIMSIZE];
                    denominator[i][j] *= other.numerator[i][MAXDIMSIZE];
                }
            } else if (name == "Mul") {
                for (uint32_t j = 0; j <= MAXDIMSIZE; ++j) {
                    numerator[i][j] *= other.numerator[i][MAXDIMSIZE];
                    denominator[i][j] *= other.denominator[i][MAXDIMSIZE];
                }
            }
        }
    }

    int64_t numerator[MAXDIMSIZE + 1][MAXDIMSIZE + 1];
    int64_t denominator[MAXDIMSIZE + 1][MAXDIMSIZE + 1];
    int64_t real_dim = -1;
    bool scalar = false;
};

struct PPLShapeOperationParam {
    std::map<edgeid_t, ShapeMatrix> alpha;
};

}}} // namespace ppl::nn::common

#endif
