#include "ppl/nn/optimizers/fuse_shape_optimizer.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/params/ppl/shape_operation_param.h"
#include "ppl/nn/params/onnx/cast_param.h"
#include "ppl/nn/params/onnx/concat_param.h"

#include <set>

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn {

inline bool IsGraphOutput(const ir::Graph* graph, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph->topo->GetOutputCount(); i++) {
        if (graph->topo->GetOutput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

bool CanFuse(ir::Node* node, PPLShapeOperationParam* shape_param, ir::Graph* graph) {
    auto& constants = graph->data->constants;
    auto topo = graph->topo.get();
    const std::set<std::string> fuse_ops{"Add", "Cast", "Concat", "Div", "Gather", "Mul", "Slice", "Sub", "Squeeze", "Unsqueeze"};
    if (!node || node->GetType().domain != "" || 
        fuse_ops.find(node->GetType().name) == fuse_ops.end()) {
        return false;
    }

    // all node's inputs can not be graph output
    for (uint32_t i = 0; i< node->GetInputCount(); ++i) {
        auto edge_id = node->GetInput(i);
        auto edge = topo->GetEdgeById(edge_id);
        if (edge_id == INVALID_EDGEID || topo->GetOutput(edge->GetName()) != INVALID_EDGEID) {
            return false;
        }
    }

    // all node's inputs must be one of Shape op's output or Constant
    for (uint32_t i = 0; i< node->GetInputCount(); ++i) {
        auto edge_id = node->GetInput(i);
        if (shape_param->alpha.find(edge_id) == shape_param->alpha.end() &&
            constants.find(edge_id) == constants.end()) {
            return false;
        }
    }
    return true;
}

RetCode UpdateMatrixForNextNode(ir::Node* node, std::vector<edgeid_t>* edge_array, PPLShapeOperationParam* shape_param, ir::Graph* graph) {
    auto& constants = graph->data->constants;
    auto& attributes = graph->data->attrs;
    auto& shapes = graph->data->shapes;

    auto pair = shape_param->alpha.find(node->GetInput(0));
    if (pair == shape_param->alpha.end()) {
        return RC_NOT_FOUND;
    }
    auto pre_matrix_left = pair->second;
    auto matrix = pre_matrix_left;

    // TODO: Xusi calc shape for different op
    if (node->GetType().name == "Cast") {  // Only support cast to int64 
        auto param_ref = attributes.find(node->GetId());
        if (param_ref == attributes.end()) {
            return RC_NOT_FOUND;
        }
        auto param = *((const CastParam*)param_ref->second.get());
        if (param.to != DATATYPE_INT64) {
            return RC_UNSUPPORTED;
        }
    } else if (node->GetType().name == "Concat") {
        auto param_ref = attributes.find(node->GetId());
        if (param_ref == attributes.end()) {
            return RC_NOT_FOUND;
        }
        auto param = *((const ConcatParam*)param_ref->second.get());
        if (param.axis != 0) {
            return RC_UNSUPPORTED;
        }
        for (uint32_t i = 1; i < node->GetInputCount(); ++i) {
            auto temp_edge_id = node->GetInput(i);
            if (shape_param->alpha.find(temp_edge_id) != shape_param->alpha.end()) {
                auto ith_matrix = shape_param->alpha.find(temp_edge_id)->second;
                if (matrix.real_dim < 0 || ith_matrix.real_dim < 0) {
                    return RC_UNSUPPORTED;
                }
                matrix.Append(ith_matrix);
            } else if (constants.find(temp_edge_id) != constants.end()) {
                auto concat_input = (const int64_t*)(constants.find(temp_edge_id)->second.data.data());
                auto concat_dims = shapes.find(temp_edge_id)->second.dims;
                if (concat_dims.size() != 1 || concat_dims[0] != 1) {
                    return RC_UNSUPPORTED;
                } 
                if (matrix.real_dim < 0) {
                    return RC_UNSUPPORTED;
                }
                matrix.Append(concat_input[0]);
            } else {
                return RC_NOT_FOUND;
            }
        }
    } else if (node->GetType().name == "Gather") {
        auto indices_id = node->GetInput(1);
        auto pair = constants.find(indices_id);
        if (pair == constants.end()) {
            return RC_UNSUPPORTED;
        }
        auto gather_indices = (const int64_t*)(pair->second.data.data());
        auto& gather_dims = shapes.find(indices_id)->second.dims;
        if (gather_dims.size() != 0 && (gather_dims.size() != 1 || gather_dims[0] != 1)) {
            return RC_UNSUPPORTED;
        }
        matrix.scalar = true;
        matrix.Gather(gather_indices[0], gather_indices[0]+1);
    } else if (node->GetType().name == "Slice") { // Only support slice from head.
        // starts
        auto start_id = node->GetInput(1);
        auto start_pair = constants.find(start_id);
        if (start_pair == constants.end()) {
            return RC_UNSUPPORTED;
        }
        auto start_input = (const int64_t*)(start_pair->second.data.data());
        auto& start_dims = shapes.find(start_id)->second.dims;
        if (start_dims.size() != 0 && (start_dims.size() != 1 || start_dims[0] != 1)) {
            return RC_UNSUPPORTED;
        }
        // ends
        auto end_id = node->GetInput(2);
        auto end_pair = constants.find(end_id);
        if (end_pair == constants.end()) {
            return RC_UNSUPPORTED;
        }
        auto end_input = (const int64_t*)(end_pair->second.data.data());
        auto& end_dims = shapes.find(end_id)->second.dims;
        if (end_dims.size() != 0 && (end_dims.size() != 1 || end_dims[0] != 1)) {
            return RC_UNSUPPORTED;
        }
        if (start_input[0] < 0 || end_input[0] < 0 || end_input[0] > ppl::nn::common::ShapeMatrix::MAXDIMSIZE) {
            return RC_UNSUPPORTED;
        }
        matrix.Gather(start_input[0], end_input[0]);
    } else if (node->GetType().name == "Unsqueeze") {
        matrix.scalar = false;
    } else if (node->GetType().name == "Squeeze") {
        matrix.scalar = true;
    } else { // Arithmetic op: support add/sub constants or other matrix, but only support mul/div constants.
        auto temp_edge_id = node->GetInput(1);
        if (shape_param->alpha.find(temp_edge_id) != shape_param->alpha.end()) {
            if (node->GetType().name == "Mul" || node->GetType().name == "Div") {
                return RC_UNSUPPORTED;
            }
            auto& temp_matrix = shape_param->alpha.find(temp_edge_id)->second;
            matrix.Arithmetic(temp_matrix, node->GetType().name);
        } else if (constants.find(temp_edge_id) != constants.end()) {
            auto arith_input = (const int64_t*)(constants.find(temp_edge_id)->second.data.data());
            auto& arith_dims = shapes.find(temp_edge_id)->second.dims;
            if (arith_dims.size() != 0 && (arith_dims.size() != 1 || arith_dims[0] != 1)) {
                return RC_UNSUPPORTED;
            }
            for (uint32_t i = 0; i < matrix.real_dim; ++i) {  // jump if shape has been divided
                if (matrix.denominator[i][ppl::nn::common::ShapeMatrix::MAXDIMSIZE] != 1) {
                    return RC_UNSUPPORTED;
                }
            }
            ShapeMatrix temp_matrix;
            temp_matrix.real_dim = 0;
            for (int32_t j = 0; j < matrix.real_dim; ++j) {
                temp_matrix.Append(arith_input[j]);
            }
            matrix.Arithmetic(temp_matrix, node->GetType().name);
        } else {
            return RC_NOT_FOUND;
        }
    }

    // load results
    auto edge_id = node->GetOutput(0);
    edge_array->push_back(edge_id);
    shape_param->alpha.emplace(edge_id, matrix);

    // delete node and all its input edge (avoid delete one more times)
    for (uint32_t i = 0; i< node->GetInputCount(); ++i) {
        auto input_edge_id = node->GetInput(i);
        if (input_edge_id == INVALID_EDGEID) {
            continue;
        }
        if (constants.find(input_edge_id) != constants.end()) {
            auto constants_edge = graph->topo->GetEdgeById(input_edge_id);
            constants_edge->DelConsumer(node->GetId());
            if (constants_edge->CalcConsumerCount() == 0) {
                constants.erase(input_edge_id);
            }
        }
    }

    graph->topo->DelNodeById(node->GetId());
    return RC_SUCCESS;
}

void FuseNextNode(edgeid_t edge_id, std::vector<edgeid_t>* edge_array, PPLShapeOperationParam* shape_param, ir::Graph* graph) {
    auto& topo = graph->topo;
    auto edge = topo->GetEdgeById(edge_id);
    for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        auto next_node_id = it.Get();
        auto next_node = topo->GetNodeById(next_node_id);
        if (next_node == nullptr) {
            continue;
        }
        if (next_node && CanFuse(next_node, shape_param, graph)) {
            UpdateMatrixForNextNode(next_node, edge_array, shape_param, graph);
        }
    }
    return;
}

void UpdateGraph(ir::Node* node, PPLShapeOperationParam* shape_param, ir::Graph* graph) {
    auto topo = graph->topo.get();
    bool first_add = true;
    for (auto it = shape_param->alpha.begin(); it != shape_param->alpha.end(); ++it) {
        auto edge = topo->GetEdgeById(it->first);
        bool flag = false;
        for (auto edge_it = edge->CreateConsumerIter(); edge_it.IsValid();) {
            auto temp_node_id = edge_it.Get();
            if (topo->GetNodeById(temp_node_id) != nullptr) {
                flag = true;
                edge_it.Forward();
            } else {
                edge->DelConsumer(temp_node_id);
            }
        }
        if (flag) {
            edge->SetProducer(node->GetId());
            if (first_add) {
                first_add = false;
                node->ReplaceOutput(node->GetOutput(0), it->first);
            } else {
                node->AddOutput(it->first);
            }
        } else {
            topo->DelEdgeById(it->first);
        }
    }
    LOG(DEBUG) << "Output count " << node->GetOutputCount() << " for fused shape node[" << node->GetName();
    return;
}

// fuse shape and its following ops
RetCode FuseShapeOptimizer::Optimize(ir::Graph* graph) const {
    for (auto node_it = graph->topo->CreateNodeIter(); node_it->IsValid(); node_it->Forward()) {
        auto node = node_it->Get();
        if (!node || node->GetType().domain != "" || node->GetType().name != "Shape") {
            continue;
        }
        node->SetType(ir::Node::Type{"ppl", "Shape"});
        node->SetName(node->GetName() + "_Fused");

        PPLShapeOperationParam shape_param;
        ShapeMatrix temp_matrix;
        shape_param.alpha.emplace(node->GetOutput(0), temp_matrix);
        std::vector<edgeid_t> edge_array{node->GetOutput(0)};
        for (uint32_t i = 0; i < edge_array.size(); ++i) {
            auto edge_id = edge_array[i];
            if (IsGraphOutput(graph, edge_id)) {
                continue;
            }
            FuseNextNode(edge_id, &edge_array, &shape_param, graph);
        }
        shared_ptr<PPLShapeOperationParam> sp = make_shared<PPLShapeOperationParam>(shape_param);
        graph->data->attrs.emplace(node->GetId(), std::move(sp));
        UpdateGraph(node, &shape_param, graph);
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
