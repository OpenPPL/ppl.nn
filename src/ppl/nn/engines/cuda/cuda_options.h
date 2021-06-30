#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_OPTIONS_H_

namespace ppl { namespace nn { namespace cuda {

enum {
    /**
       @brief set output data format

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_SET_OUTPUT_FORMAT, DATAFORMAT_NDARRAY);
       @endcode
    */
    CUDA_CONF_SET_OUTPUT_FORMAT = 0,

    /**
       @brief set output data type

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_SET_OUTPUT_TYPE, DATATYPE_FLOAT32);
       @endcode
    */
    CUDA_CONF_SET_OUTPUT_TYPE,

    /**
       @brief set init input dims for compiler

       @note example:
       @code{.cpp}
       std::string dims = "1,3,224,224"
       engine->Configure(CUDA_CONF_SET_COMPILER_INPUT_SHAPE, dims.c_str);
       @endcode
    */
    CUDA_CONF_SET_COMPILER_INPUT_SHAPE,

    /**
       @brief set kernel default execution data type

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_SET_KERNEL_DEFAULT_TYPE, DATATYPE_FLOAT32);
       @endcode
    */
    CUDA_CONF_SET_KERNEL_DEFAULT_TYPE,

    /**
       @brief use default algorithms for conv and gemm

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_USE_DEFAULT_ALGORITHMS, true);
       @endcode
    */
    CUDA_CONF_SET_DEFAULT_ALGORITHMS,

    /**
       @brief set node execution type

       @note example:
       @code{.cpp}
       std::string node_types = "Conv_0,DATATYPE_FLOAT32;Conv_1,DATATYPE_FLOAT16"
       cuda_engine->Configure(CUDA_CONF_SET_NODE_DATA_TYPE, node_types.c_str());
       @endcode
    */
    CUDA_CONF_SET_NODE_DATA_TYPE,

    /** max value */
    CUDA_CONF_MAX,
};

}}} // namespace ppl::nn::cuda

#endif
