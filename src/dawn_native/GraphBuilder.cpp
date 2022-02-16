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

#include "dawn_native/GraphBuilder.h"

#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/Assert.h"
#include "common/Log.h"
#include "common/RefCounted.h"
#include "dawn_native/Graph.h"
#include "dawn_native/NamedOperands.h"
#include "dawn_native/Operand.h"
#include "dawn_native/Operator.h"
#include "dawn_native/ops/Binary.h"
#include "dawn_native/ops/Conv2d.h"
#include "dawn_native/ops/Constant.h"
#include "dawn_native/ops/Input.h"
#include "dawn_native/ops/Unary.h"

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

    OperandBase* GraphBuilderBase::APIConv2d(OperandBase* input,
                                             OperandBase* filter,
                                             Conv2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Conv2d(this, input, filter, options));
    }

    OperandBase* GraphBuilderBase::APIRelu(OperandBase* x) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kRelu, x));
    }

    FusionOperatorBase* GraphBuilderBase::APIReluOperator() {
        return new op::FusionUnary(this, FusionType::Relu);
    }

    NamedOperandsBase* GraphBuilderBase::APICreateNamedOperands() {
        return new NamedOperandsBase();
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
        Ref<GraphBase> graph = AcquireRef(CreateGraphImpl());
        for (auto& op : sorted_operands) {
            if (op->IsError() || GetDevice()->ConsumedError(op->AddToGraph(graph.Get()))) {
                dawn::ErrorLog() << "Failed to add the operand when building graph.";
                return nullptr;
            }
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            if (GetDevice()->ConsumedError(
                    graph->AddOutput(namedOutput.first, namedOutput.second))) {
                dawn::ErrorLog() << "Failed to add output when building graph.";
                return nullptr;
            }
        }
        if (GetDevice()->ConsumedError(graph->Finish())) {
            dawn::ErrorLog() << "Failed to finish building graph.";
            return nullptr;
        }

        if (GetDevice()->ConsumedError(graph->Compile())) {
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

    bool GraphBuilderBase::InitializeImpl() {
        dawn::InfoLog() << "Unimplemented: GraphBuilderBase::InitializeImpl()";
        return true;
    }

    GraphBase* GraphBuilderBase::CreateGraphImpl() {
        dawn::InfoLog() << "Unimplemented: GraphBuilderBase::CreateGraphImpl()";
        return new GraphBase(GetDevice());
    }

}  // namespace webnn_native
