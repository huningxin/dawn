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

#ifndef WEBNN_NATIVE_GRAPH_H_
#define WEBNN_NATIVE_GRAPH_H_

#include "dawn/common/RefCounted.h"
#include "dawn/native/Error.h"
#include "dawn/native/Forward.h"
#include "dawn/native/GraphBuilder.h"
#include "dawn/native/ObjectBase.h"
#include "dawn/native/Operand.h"
#include "dawn/native/dawn_platform.h"

namespace dawn::native {

    namespace op {
        class Constant;
        class Input;
        class BatchNorm;
        class Binary;
        class Conv2d;
        class Gru;
        class Pad;
        class Pool2d;
        class Reduce;
        class Resample2d;
        class Reshape;
        class Slice;
        class Split;
        class Squeeze;
        class Transpose;
        class Unary;
        class LeakyRelu;
        class Concat;
        class Gemm;
        class Clamp;
        class InstanceNorm;
    }  // namespace op

    class GraphBase : public ObjectBase {
      public:
        static GraphBase* MakeError(DeviceBase* device);

        explicit GraphBase(DeviceBase* device);
        virtual ~GraphBase() = default;

        virtual MaybeError AddConstant(const op::Constant* constant);
        virtual MaybeError AddInput(const op::Input* input);
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* output);
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm);
        virtual MaybeError AddBinary(const op::Binary* binary);
        virtual MaybeError AddClamp(const op::Clamp* clamp);
        virtual MaybeError AddConcat(const op::Concat* concat);
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d);
        virtual MaybeError AddGemm(const op::Gemm* gemm);
        virtual MaybeError AddPad(const op::Pad* pad);
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d);
        virtual MaybeError AddReduce(const op::Reduce* reduce);
        virtual MaybeError AddResample2d(const op::Resample2d* resample2d);
        virtual MaybeError AddReshape(const op::Reshape* reshape);
        virtual MaybeError AddTranspose(const op::Transpose* transpose);
        virtual MaybeError AddUnary(const op::Unary* unary);
        virtual MaybeError Finish();
        virtual MaybeError Compile();

        // Webnn API
        void APICompute(NamedResourcesBase* inputs, NamedResourcesBase* outputs);
        NamedResourcesBase* APICreateNamedResources();

      private:
        GraphBase(DeviceBase* device, ObjectBase::ErrorTag tag);
        virtual MaybeError CompileImpl();
        virtual void ComputeImpl(NamedResourcesBase* inputs,
                                 NamedResourcesBase* outputs);
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_MODEL_H_
