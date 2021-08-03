#ifndef _ST_HPC_PPL_NN_PARAMS_PPL_SHAPE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PPL_SHAPE_PARAM_H_

#include "ppl/nn/common/logger.h"
#include "ppl/nn/common/types.h"

#include <stdint.h>

#include <vector>
#include <string>

namespace ppl { namespace nn { namespace common {

struct Matrix {
    static const int64_t MAXDIMSIZE = 8;

    Matrix() {
        for (int32_t i = 0; i <= MAXDIMSIZE; ++i) {
            for (int32_t j = 0; j <= MAXDIMSIZE; ++j) {
                matrix_2d[i][j] = (i == j ? 1 : 0);
            }
        }
    }


    void Append(const Matrix& other) {
        if (real_dim < 0 || other.real_dim < 0) {
            LOG(ERROR) << "Matrix has unknown dim count and cannot append.";
            return;
        }
        for (int32_t i = 0; i < other.real_dim; ++i) {
            for (int32_t j = 0; j <= MAXDIMSIZE; ++j) {
                matrix_2d[i + real_dim][j] = other.matrix_2d[i][j];
            }
        }
        real_dim += other.real_dim;
    }

    void Append(int64_t value) {
        if (real_dim < 0) {
            LOG(ERROR) << "Matrix has unknown dim count and cannot append.";
            return;
        }
        for (int32_t j = 0; j < MAXDIMSIZE; ++j) {
            matrix_2d[real_dim][j] = 0;
        }
        matrix_2d[real_dim][MAXDIMSIZE] = value;
        real_dim += 1;
    }

    void Gather(int64_t begin, int64_t end) {
        real_dim = end - begin;
        for (int32_t i = 0; i < real_dim; ++i) {
            for (int32_t j = 0; j < MAXDIMSIZE; ++j) {
                matrix_2d[i][j] = matrix_2d[begin + i][j];
            }
        }
    }

    void Arithmetic(const Matrix& other, std::string name) {
        if (other.real_dim != real_dim) {
            LOG(ERROR) << "Matrix has incorrect dim count for arithmetic op.";
            return;
        }
        for (int32_t i = 0; i < real_dim; ++i) {
            if (name == "Add") {
                matrix_2d[i][MAXDIMSIZE] = other.matrix_2d[i][MAXDIMSIZE];
            } else if (name == "Sub") {
                matrix_2d[i][MAXDIMSIZE] = -other.matrix_2d[i][MAXDIMSIZE];
            } else if (name == "Div") {
                for (int32_t j = 0; j <= MAXDIMSIZE; ++j) {
                    matrix_2d[i][j] = matrix_2d[i][j] / other.matrix_2d[i][MAXDIMSIZE];
                }
            } else if (name == "Mul") {
                for (uint32_t j = 0; j <= MAXDIMSIZE; ++j) {
                    matrix_2d[i][j] = matrix_2d[i][j] * other.matrix_2d[i][MAXDIMSIZE];
                }
            }
        }
    }

    double matrix_2d[MAXDIMSIZE + 1][MAXDIMSIZE + 1];
    int real_dim = -1;
    bool scalar = false;
};

struct PPLShapeParam {
    std::map<edgeid_t, Matrix> alpha;
};

}}} // namespace ppl::nn::common

#endif
