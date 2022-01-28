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

#ifndef WEBNN_NATIVE_DML_GRAPH_DML_H_
#define WEBNN_NATIVE_DML_GRAPH_DML_H_

#include <map>
#include <mutex>
#include <set>
#include <unordered_set>

#include "dawn_native/Graph.h"
#include "dawn_native/Operand.h"
#include "dawn_native/Operator.h"
#include "dawn_native/dml/deps/src/precomp.h"
#include "dawn_native/ops/Constant.h"
#include "dawn_native/ops/Input.h"
#include "dawn_native/ops/Unary.h"

namespace dawn::native { namespace dml {

    std::string DmlTensorDimensionsToString(const ::dml::TensorDimensions&);
    std::string DmlTensorDataTypeToString(DML_TENSOR_DATA_TYPE type);

    class Graph : public GraphBase {
      public:
        explicit Graph(DeviceBase* device);
        ~Graph() override = default;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* output) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError Finish() override;

      private:
        MaybeError CompileImpl() override;
        void ComputeImpl(NamedResourcesBase* inputs,
                         NamedResourcesBase* outputs) override;

        ::dml::Expression BindingConstant(DML_TENSOR_DATA_TYPE dmlTensorType,
                                          ::dml::TensorDimensions dmlTensorDims,
                                          BufferBase* buffer,
                                          size_t offset,
                                          size_t size);

        std::shared_ptr<::pydml::Device> mDevice;
        // The mutex is used to lock mDevice.
        std::mutex mMutex;
        std::unique_ptr<::dml::Graph> mGraph;
        std::map<const OperandBase*, ::dml::Expression> mExpression;
        std::vector<std::unique_ptr<::pydml::Binding>> mInputBindings;
        std::vector<std::unique_ptr<::pydml::Binding>> mOutputBindings;
        std::unordered_set<const OperandBase*> mConstantSet;
        std::vector<::dml::Expression> mOutputExpressions;
        std::map<std::string, ::pydml::Binding*> mInputs;
        std::map<std::string, ::pydml::Binding*> mOutputs;
        std::unique_ptr<pydml::CompiledModel> mCompiledModel;
    };

}}  // namespace dawn::native::dml

#endif  // WEBNN_NATIVE_DML_GRAPH_DML_H_
