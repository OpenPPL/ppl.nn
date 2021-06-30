#include "ppl/nn/ir/node.h"
#include "ppl/nn/utils/vector_utils.h"
using namespace std;

namespace ppl { namespace nn { namespace ir {

static uint32_t DoReplace(edgeid_t* vec, uint32_t size, edgeid_t old_value, edgeid_t new_value) {
    uint32_t counter = 0;
    for (uint32_t i = 0; i < size; ++i) {
        if (vec[i] == old_value) {
            vec[i] = new_value;
            ++counter;
        }
    }
    return counter;
}

uint32_t Node::ReplaceInput(edgeid_t old_value, edgeid_t new_value) {
    return DoReplace(inputs_.data(), inputs_.size(), old_value, new_value);
}

void Node::AddOutput(edgeid_t eid) {
    utils::VectorAddUnique(outputs_, eid);
}

uint32_t Node::ReplaceOutput(edgeid_t old_value, edgeid_t new_value) {
    return DoReplace(outputs_.data(), outputs_.size(), old_value, new_value);
}

void Node::AddExtraInput(edgeid_t eid) {
    utils::VectorAddUnique(extra_inputs_, eid);
}

uint32_t Node::ReplaceExtraInput(edgeid_t old_value, edgeid_t new_value) {
    return DoReplace(extra_inputs_.data(), extra_inputs_.size(), old_value, new_value);
}

}}} // namespace ppl::nn::ir
