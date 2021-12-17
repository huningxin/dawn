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

#include "webnn/webnn_native/GraphBuilder.h"

#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/Assert.h"
#include "common/Log.h"
#include "common/RefCounted.h"
#include "webnn/webnn_native/Context.h"
#include "webnn/webnn_native/Graph.h"
#include "webnn/webnn_native/Operand.h"
#include "webnn/webnn_native/OperandArray.h"
#include "webnn/webnn_native/Operator.h"
#include "webnn/webnn_native/ops/BatchNorm.h"
#include "webnn/webnn_native/ops/Binary.h"
#include "webnn/webnn_native/ops/Clamp.h"
#include "webnn/webnn_native/ops/Concat.h"
#include "webnn/webnn_native/ops/Constant.h"
#include "webnn/webnn_native/ops/Conv2d.h"
#include "webnn/webnn_native/ops/Gemm.h"
#include "webnn/webnn_native/ops/Gru.h"
#include "webnn/webnn_native/ops/Input.h"
#include "webnn/webnn_native/ops/InstanceNorm.h"
#include "webnn/webnn_native/ops/LeakyRelu.h"
#include "webnn/webnn_native/ops/Pad.h"
#include "webnn/webnn_native/ops/Pool2d.h"
#include "webnn/webnn_native/ops/Reduce.h"
#include "webnn/webnn_native/ops/Resample.h"
#include "webnn/webnn_native/ops/Reshape.h"
#include "webnn/webnn_native/ops/Slice.h"
#include "webnn/webnn_native/ops/Split.h"
#include "webnn/webnn_native/ops/Squeeze.h"
#include "webnn/webnn_native/ops/Transpose.h"
#include "webnn/webnn_native/ops/Unary.h"

#define DAWN_VALIDATE(ptr, objectBase)                 \
    Ref<OperatorBase> op = AcquireRef(ptr);            \
    if (GetContext()->ConsumedError(op->Validate())) { \
        return objectBase::MakeError(this);            \
    }                                                  \
    for (;;)                                           \
    break

#define VALIDATE_FOR_OPERAND(ptr)    \
    DAWN_VALIDATE(ptr, OperandBase); \
    return op->PrimaryOutput()
#define VALIDATE_FUSED_OPERATOR(ptr)  \
    DAWN_VALIDATE(ptr, OperatorBase); \
    return op.Detach()
#define VALIDATE_ARRAY_OPERAND(ptr)       \
    DAWN_VALIDATE(ptr, OperandArrayBase); \
    return new OperandArrayBase(this, op->Outputs())

namespace webnn_native {

    GraphBuilderBase::GraphBuilderBase(ContextBase* context) : ObjectBase(context) {
    }

    OperandBase* GraphBuilderBase::APIAbs(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kAbs, input));
    }

    OperandBase* GraphBuilderBase::APIAdd(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kAdd, a, b));
    }

    OperandBase* GraphBuilderBase::APIAveragePool2d(OperandBase* input,
                                                    Pool2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pool2d(this, op::Pool2dType::kAveragePool2d, input, options));
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

    OperatorBase* GraphBuilderBase::APIClampOperator(ClampOptions const* options) {
        VALIDATE_FUSED_OPERATOR(new op::Clamp(this, options));
    }

    OperandBase* GraphBuilderBase::APICeil(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kCeil, input));
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

    OperandBase* GraphBuilderBase::APIConstant(OperandDescriptor const* desc,
                                               ArrayBufferView const* arrayBuffer) {
        VALIDATE_FOR_OPERAND(new op::Constant(this, desc, arrayBuffer));
    }

    OperandBase* GraphBuilderBase::APIConv2d(OperandBase* input,
                                             OperandBase* filter,
                                             Conv2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Conv2d(this, input, filter, options));
    }

    OperandBase* GraphBuilderBase::APICos(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kCos, input));
    }

    OperandBase* GraphBuilderBase::APIDiv(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kDiv, a, b));
    }

    OperandBase* GraphBuilderBase::APIExp(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kExp, input));
    }

    OperandBase* GraphBuilderBase::APIFloor(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kFloor, input));
    }

    OperandBase* GraphBuilderBase::APIGemm(OperandBase* a,
                                           OperandBase* b,
                                           GemmOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Gemm(this, a, b, options));
    }

    OperandArrayBase* GraphBuilderBase::APIGru(OperandBase* input,
                                               OperandBase* weight,
                                               OperandBase* recurrentWeight,
                                               int32_t steps,
                                               int32_t hiddenSize,
                                               GruOptions const* options) {
        VALIDATE_ARRAY_OPERAND(
            new op::Gru(this, input, weight, recurrentWeight, steps, hiddenSize, options));
    }

    OperandBase* GraphBuilderBase::APIHardSwish(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kHardSwish, input));
    }

    OperatorBase* GraphBuilderBase::APIHardSwishOperator() {
        VALIDATE_FUSED_OPERATOR(
            new op::Unary(this, op::UnaryOpType::kHardSwish, FusedOperator::HardSwish));
    }

    OperandBase* GraphBuilderBase::APIInput(char const* name, OperandDescriptor const* desc) {
        VALIDATE_FOR_OPERAND(new op::Input(this, std::string(name), desc));
    }

    OperandBase* GraphBuilderBase::APIInstanceNorm(OperandBase* input,
                                                   InstanceNormOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::InstanceNorm(this, input, options));
    }

    OperandBase* GraphBuilderBase::APILeakyRelu(OperandBase* input,
                                                LeakyReluOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::LeakyRelu(this, input, options));
    }

    OperatorBase* GraphBuilderBase::APILeakyReluOperator(LeakyReluOptions const* options) {
        VALIDATE_FUSED_OPERATOR(new op::LeakyRelu(this, options));
    }

    OperandBase* GraphBuilderBase::APILog(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kLog, input));
    }

    OperandBase* GraphBuilderBase::APIMatmul(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMatMul, a, b));
    }

    OperandBase* GraphBuilderBase::APIMax(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMax, a, b));
    }

    OperandBase* GraphBuilderBase::APIMaxPool2d(OperandBase* input, Pool2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pool2d(this, op::Pool2dType::kMaxPool2d, input, options));
    }

    OperandBase* GraphBuilderBase::APIMin(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMin, a, b));
    }

    OperandBase* GraphBuilderBase::APIMul(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMul, a, b));
    }

    OperandBase* GraphBuilderBase::APINeg(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kNeg, input));
    }

    OperandBase* GraphBuilderBase::APIPad(OperandBase* input,
                                          OperandBase* padding,
                                          PadOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pad(this, input, padding, options));
    }

    OperandBase* GraphBuilderBase::APIPow(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kPower, a, b));
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

    OperandBase* GraphBuilderBase::APIReduceProduct(OperandBase* input,
                                                    ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceProduct, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceSum(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceSum, input, options));
    }

    OperandBase* GraphBuilderBase::APIRelu(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kRelu, input));
    }

    OperatorBase* GraphBuilderBase::APIReluOperator() {
        VALIDATE_FUSED_OPERATOR(new op::Unary(this, op::UnaryOpType::kRelu, FusedOperator::Relu));
    }

    OperandBase* GraphBuilderBase::APIResample(OperandBase* input, ResampleOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Resample(this, input, options));
    }

    OperandBase* GraphBuilderBase::APIReshape(OperandBase* input,
                                              int32_t const* new_shape,
                                              size_t new_shape_count) {
        VALIDATE_FOR_OPERAND(new op::Reshape(this, input, new_shape, new_shape_count));
    }

    OperandBase* GraphBuilderBase::APISigmoid(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kSigmoid, input));
    }

    OperatorBase* GraphBuilderBase::APISigmoidOperator() {
        VALIDATE_FUSED_OPERATOR(
            new op::Unary(this, op::UnaryOpType::kSigmoid, FusedOperator::Sigmoid));
    }

    OperandBase* GraphBuilderBase::APISin(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kSin, input));
    }

    OperandBase* GraphBuilderBase::APISlice(OperandBase* input,
                                            int32_t const* starts,
                                            uint32_t startsCount,
                                            int32_t const* sizes,
                                            uint32_t sizesCount,
                                            SliceOptions const* options) {
        VALIDATE_FOR_OPERAND(
            new op::Slice(this, input, starts, startsCount, sizes, sizesCount, options));
    }

    OperandBase* GraphBuilderBase::APISoftmax(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kSoftmax, input));
    }

    OperandArrayBase* GraphBuilderBase::APISplit(OperandBase* input,
                                                 uint32_t const* splits,
                                                 uint32_t splitsCount,
                                                 SplitOptions const* options) {
        VALIDATE_ARRAY_OPERAND(new op::Split(this, input, splits, splitsCount, options));
    }

    OperandBase* GraphBuilderBase::APISqueeze(OperandBase* input, SqueezeOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Squeeze(this, input, options));
    }

    OperandBase* GraphBuilderBase::APISub(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kSub, a, b));
    }

    OperandBase* GraphBuilderBase::APITan(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kTan, input));
    }

    OperandBase* GraphBuilderBase::APITanh(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kTanh, input));
    }

    OperatorBase* GraphBuilderBase::APITanhOperator() {
        VALIDATE_FUSED_OPERATOR(new op::Unary(this, op::UnaryOpType::kTanh, FusedOperator::Tanh));
    }

    OperandBase* GraphBuilderBase::APITranspose(OperandBase* input,
                                                TransposeOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Transpose(this, input, options));
    }

    GraphBase* GraphBuilderBase::APIBuild(NamedOperandsBase const* namedOperands) {
        if (DAWN_UNLIKELY(this->IsError())) {
            dawn::ErrorLog() << "This Graph object is an error";
            return nullptr;
        }

        std::vector<const OperandBase*> outputs;
        if (namedOperands->GetRecords().empty()) {
            dawn::ErrorLog() << "The output named operands are empty.";
            return nullptr;
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            outputs.push_back(namedOutput.second);
        }
        std::vector<const OperatorBase*> sorted_operands = TopologicalSort(outputs);
        Ref<GraphBase> graph = AcquireRef(GetContext()->CreateGraph());
        for (auto& op : sorted_operands) {
            if (op->IsError() || GetContext()->ConsumedError(op->AddToGraph(graph.Get()))) {
                dawn::ErrorLog() << "Failed to add the operand when building graph.";
                return nullptr;
            }
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            if (GetContext()->ConsumedError(
                    graph->AddOutput(namedOutput.first, namedOutput.second))) {
                dawn::ErrorLog() << "Failed to add output when building graph.";
                return nullptr;
            }
        }
        if (GetContext()->ConsumedError(graph->Finish())) {
            dawn::ErrorLog() << "Failed to finish building graph.";
            return nullptr;
        }

        if (GetContext()->ConsumedError(graph->Compile())) {
            dawn::ErrorLog() << "Failed to compile the graph.";
            return nullptr;
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
            nodesToDo.push(const_cast<OperandBase*>(node)->Operator());
        }
        while (nodesToDo.size() > 0) {
            const OperatorBase* node = nodesToDo.top();
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

}  // namespace webnn_native
