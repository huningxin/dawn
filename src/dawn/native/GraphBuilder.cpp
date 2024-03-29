// Copyright 2021 The WebNN-native Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dawn/native/GraphBuilder.h"

#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include "dawn/common/Assert.h"
#include "dawn/common/Log.h"
#include "dawn/common/RefCounted.h"
#include "dawn/native/Graph.h"
#include "dawn/native/NamedOperands.h"
#include "dawn/native/Operand.h"
#include "dawn/native/Operator.h"
#include "dawn/native/ops/BatchNorm.h"
#include "dawn/native/ops/Binary.h"
#include "dawn/native/ops/Clamp.h"
#include "dawn/native/ops/Concat.h"
#include "dawn/native/ops/Conv2d.h"
#include "dawn/native/ops/Constant.h"
#include "dawn/native/ops/Gemm.h"
#include "dawn/native/ops/LeakyRelu.h"
#include "dawn/native/ops/Input.h"
#include "dawn/native/ops/Pad.h"
#include "dawn/native/ops/Pool2d.h"
#include "dawn/native/ops/Reduce.h"
#include "dawn/native/ops/Resample2d.h"
#include "dawn/native/ops/Reshape.h"
#include "dawn/native/ops/Transpose.h"
#include "dawn/native/ops/Unary.h"

#define WEBNN_VALIDATE(ptr, objectBase)                                  \
    Ref<OperatorBase> op = AcquireRef(ptr);                              \
    if (GetDevice()->ConsumedError(op->ValidateAndInferOutputInfo())) { \
        return objectBase::MakeError(this);                              \
    }                                                                    \
    for (;;)                                                             \
    break

#define VALIDATE_FOR_OPERAND(ptr)     \
    WEBNN_VALIDATE(ptr, OperandBase); \
    return op->PrimaryOutput()
#define VALIDATE_ARRAY_OPERAND(ptr)        \
    WEBNN_VALIDATE(ptr, OperandArrayBase); \
    return new OperandArrayBase(this, op->Outputs())

namespace dawn::native {

    // static
    GraphBuilderBase* GraphBuilderBase::Create(DeviceBase* device) {
        return new GraphBuilderBase(device);
    }

    GraphBuilderBase* GraphBuilderBase::MakeError(DeviceBase* device) {
        return new GraphBuilderBase(device, ObjectBase::kError);
    }

    bool GraphBuilderBase::Initialize() {
        return InitializeImpl();
    }

    GraphBuilderBase::GraphBuilderBase(DeviceBase* device) : ObjectBase(device) {
    }

    GraphBuilderBase::GraphBuilderBase(DeviceBase* device, ObjectBase::ErrorTag tag)
        : ObjectBase(device, tag) {
    }

    OperandBase* GraphBuilderBase::APIConstant(OperandDescriptor const* desc,
                                               BufferResourceView const* view) {
        VALIDATE_FOR_OPERAND(new op::Constant(this, desc, view));
    }

    OperandBase* GraphBuilderBase::APIInput(char const* name, OperandDescriptor const* desc) {
        VALIDATE_FOR_OPERAND(new op::Input(this, std::string(name), desc));
    }

    OperandBase* GraphBuilderBase::APIAdd(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kAdd, a, b));
    }

    OperandBase* GraphBuilderBase::APIDiv(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kDiv, a, b));
    }

    OperandBase* GraphBuilderBase::APIMul(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMul, a, b));
    }

    OperandBase* GraphBuilderBase::APISub(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kSub, a, b));
    }

    OperandBase* GraphBuilderBase::APIMax(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMax, a, b));
    }

    OperandBase* GraphBuilderBase::APIMin(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMin, a, b));
    }

    OperandBase* GraphBuilderBase::APIPow(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kPower, a, b));
    }

    OperandBase* GraphBuilderBase::APIBatchNorm(OperandBase* input,
                                                OperandBase* mean,
                                                OperandBase* variance,
                                                BatchNormOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::BatchNorm(this, input, mean, variance, options));
    }

    OperandBase* GraphBuilderBase::APIClamp(OperandBase* input, ClampOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Clamp(this, input, options));
    }

    FusionOperatorBase* GraphBuilderBase::APIClampOperator(ClampOptions const* options) {
        return new op::FusionClamp(this, options);
    }

    OperandBase* GraphBuilderBase::APIConcat(uint32_t inputsCount,
                                             OperandBase* const* inputs,
                                             uint32_t axis) {
        std::vector<Ref<OperandBase>> operandInputs;
        operandInputs.reserve(inputsCount);
        for (uint32_t i = 0; i < inputsCount; ++i) {
            operandInputs.push_back(inputs[i]);
        }
        VALIDATE_FOR_OPERAND(new op::Concat(this, std::move(operandInputs), axis));
    }

    OperandBase* GraphBuilderBase::APIConv2d(OperandBase* input,
                                             OperandBase* filter,
                                             Conv2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Conv2d(this, input, filter, options));
    }

    OperandBase* GraphBuilderBase::APIGemm(OperandBase* a,
                                           OperandBase* b,
                                           GemmOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Gemm(this, a, b, options));
    }

    OperandBase* GraphBuilderBase::APILeakyRelu(OperandBase* input, LeakyReluOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::LeakyRelu(this, input, options));
    }

    FusionOperatorBase* GraphBuilderBase::APILeakyReluOperator(LeakyReluOptions const* options) {
        return new op::FusionLeakyRelu(this, options);
    }

    OperandBase* GraphBuilderBase::APIMatmul(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMatMul, a, b));
    }

    OperandBase* GraphBuilderBase::APIAveragePool2d(OperandBase* input, Pool2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pool2d(this, op::Pool2dType::kAveragePool2d, input, options));
    }

    OperandBase* GraphBuilderBase::APIMaxPool2d(OperandBase* input, Pool2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pool2d(this, op::Pool2dType::kMaxPool2d, input, options));
    }

    OperandBase* GraphBuilderBase::APIPad(OperandBase* input,
                                          uint32_t const* padding,
                                          size_t padding_count,
                                          PadOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pad(this, input, padding, padding_count, options));
    }

    OperandBase* GraphBuilderBase::APIRelu(OperandBase* x) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kRelu, x));
    }

    FusionOperatorBase* GraphBuilderBase::APIReluOperator() {
        return new op::FusionUnary(this, FusionType::Relu);
    }

    OperandBase* GraphBuilderBase::APIReduceArgMax(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceArgMax, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceArgMin(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceArgMin, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceL2(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceL2, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceL1(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceL1, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceMax(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceMax, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceMean(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceMean, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceMin(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceMin, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceProduct(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceProduct, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceSum(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceSum, input, options));
    }

    OperandBase* GraphBuilderBase::APIResample2d(OperandBase* input,
                                                 Resample2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Resample2d(this, input, options));
    }

    OperandBase* GraphBuilderBase::APIReshape(OperandBase* input,
                                              int32_t const* new_shape,
                                              size_t new_shape_count) {
        VALIDATE_FOR_OPERAND(new op::Reshape(this, input, new_shape, new_shape_count));
    }

    OperandBase* GraphBuilderBase::APISigmoid(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kSigmoid, input));
    }

    FusionOperatorBase* GraphBuilderBase::APISigmoidOperator() {
        return new op::FusionUnary(this, FusionType::Sigmoid);
    }

    OperandBase* GraphBuilderBase::APISoftmax(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kSoftmax, input));
    }

    OperandBase* GraphBuilderBase::APITranspose(OperandBase* input, TransposeOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Transpose(this, input, options));
    }

    NamedOperandsBase* GraphBuilderBase::APICreateNamedOperands() {
        return new NamedOperandsBase();
    }

    GraphBase* GraphBuilderBase::APIBuild(NamedOperandsBase const* namedOperands) {
        if (DAWN_UNLIKELY(this->IsError())) {
            dawn::ErrorLog() << "This GraphBuilder object is an error";
            return GraphBase::MakeError(GetDevice());
        }

        std::vector<const OperandBase*> outputs;
        if (namedOperands->GetRecords().empty()) {
            dawn::ErrorLog() << "The output named operands are empty.";
            return GraphBase::MakeError(GetDevice());
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            outputs.push_back(namedOutput.second);
        }
        std::vector<const OperatorBase*> sorted_operands = TopologicalSort(outputs);
        if (sorted_operands.empty()) {
            dawn::ErrorLog() << "Failed to sort graph.";
            return GraphBase::MakeError(GetDevice());
        }
        Ref<GraphBase> graph = AcquireRef(CreateGraphImpl());
        for (auto& op : sorted_operands) {
            if (op->IsError() || GetDevice()->ConsumedError(op->AddToGraph(graph.Get()))) {
                dawn::ErrorLog() << "Failed to add the operand when building graph.";
                return GraphBase::MakeError(GetDevice());
            }
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            if (GetDevice()->ConsumedError(
                    graph->AddOutput(namedOutput.first, namedOutput.second))) {
                dawn::ErrorLog() << "Failed to add output when building graph.";
                return GraphBase::MakeError(GetDevice());
            }
        }
        if (GetDevice()->ConsumedError(graph->Finish())) {
            dawn::ErrorLog() << "Failed to finish building graph.";
            return GraphBase::MakeError(GetDevice());
        }

        if (GetDevice()->ConsumedError(graph->Compile())) {
            dawn::ErrorLog() << "Failed to compile the graph.";
            return GraphBase::MakeError(GetDevice());
        }

        return graph.Detach();
    }

    // The implementation derives from nGraph topological_sort in
    // https://github.com/openvinotoolkit/openvino/blob/master/ngraph/core/include/ngraph/graph_util.hpp
    //
    //*****************************************************************************
    // Copyright 2017-2020 Intel Corporation
    //
    // Licensed under the Apache License, Version 2.0 (the "License");
    // you may not use this file except in compliance with the License.
    // You may obtain a copy of the License at
    //
    //     http://www.apache.org/licenses/LICENSE-2.0
    //
    // Unless required by applicable law or agreed to in writing, software
    // distributed under the License is distributed on an "AS IS" BASIS,
    // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    // See the License for the specific language governing permissions and
    // limitations under the License.
    //*****************************************************************************
    std::vector<const OperatorBase*> GraphBuilderBase::TopologicalSort(
        std::vector<const OperandBase*>& rootNodes) {
        std::stack<const OperatorBase*> nodesToDo;
        std::unordered_set<const OperatorBase*> nodesDone;
        std::vector<const OperatorBase*> result;

        for (auto node : rootNodes) {
            if (node->IsError()) {
                return {};
            }
            nodesToDo.push(const_cast<OperandBase*>(node)->Operator());
        }
        while (nodesToDo.size() > 0) {
            const OperatorBase* node = nodesToDo.top();
            if (node->IsError()) {
                return {};
            }
            if (nodesDone.count(node) == 0) {
                bool can_add = true;
                for (auto& dep : node->Inputs()) {
                    if (nodesDone.count(dep->Operator()) == 0) {
                        can_add = false;
                        nodesToDo.push(dep->Operator());
                    }
                }
                if (can_add) {
                    result.push_back(node);
                    nodesToDo.pop();
                    nodesDone.insert(node);
                }
            } else {
                nodesToDo.pop();
            }
        }
        return result;
    }

    bool GraphBuilderBase::InitializeImpl() {
        dawn::InfoLog() << "Unimplemented: GraphBuilderBase::InitializeImpl()";
        return true;
    }

    GraphBase* GraphBuilderBase::CreateGraphImpl() {
        dawn::InfoLog() << "Unimplemented: GraphBuilderBase::CreateGraphImpl()";
        return new GraphBase(GetDevice());
    }

}  // namespace webnn_native
