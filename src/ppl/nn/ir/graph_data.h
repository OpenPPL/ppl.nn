#ifndef _ST_HPC_PPL_NN_IR_GRAPH_DATA_H_
#define _ST_HPC_PPL_NN_IR_GRAPH_DATA_H_

#include "ppl/common/types.h"
#include <string>
#include <vector>
#include <map>

namespace ppl { namespace nn { namespace ir {

struct Shape final {
    ppl::common::datatype_t data_type;
    ppl::common::dataformat_t data_format;
    std::vector<int64_t> dims;
};

struct Constant final {
    std::string data;
};

struct GraphData final {
    std::map<edgeid_t, Constant> constants;
    std::map<edgeid_t, Shape> shapes;
    std::map<nodeid_t, std::shared_ptr<void>> attrs; // attrs can be shared with cpu engines
};

}}} // namespace ppl::nn::ir

#endif
