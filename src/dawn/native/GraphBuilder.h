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

#ifndef WEBNN_NATIVE_MODEL_BUILDER_H_
#define WEBNN_NATIVE_MODEL_BUILDER_H_

#include "dawn/common/RefCounted.h"
#include "dawn/native/Forward.h"
#include "dawn/native/Device.h"
#include "dawn/native/NamedOperands.h"
#include "dawn/native/ObjectBase.h"
#include "dawn/native/Operand.h"
#include "dawn/native/Operator.h"
#include "dawn/native/dawn_platform.h"

#include <functional>
#include <vector>

namespace dawn::native {

    class GraphBuilderBase : public ObjectBase {
      public:
        static GraphBuilderBase* Create(DeviceBase* context);

        static GraphBuilderBase* MakeError(DeviceBase* device);

        bool Initialize();

        // WebNN API
        OperandBase* APIInput(char const* name, OperandDescriptor const* desc);
        OperandBase* APIConstant(OperandDescriptor const* desc, BufferResourceView const* view);
        OperandBase* APIAdd(OperandBase* a, OperandBase* b);
        OperandBase* APIDiv(OperandBase* a, OperandBase* b);
        OperandBase* APIMul(OperandBase* a, OperandBase* b);
        OperandBase* APISub(OperandBase* a, OperandBase* b);
        OperandBase* APIMax(OperandBase* a, OperandBase* b);
        OperandBase* APIMin(OperandBase* a, OperandBase* b);
        OperandBase* APIPow(OperandBase* a, OperandBase* b);
        OperandBase* APIBatchNorm(OperandBase*,
                                  OperandBase*,
                                  OperandBase*,
                                  BatchNormOptions const* options);
        OperandBase* APIClamp(OperandBase*, ClampOptions const* options);
        FusionOperatorBase* APIClampOperator(ClampOptions const* options);
        OperandBase* APIConcat(uint32_t inputsCount, OperandBase* const* inputs, uint32_t axis);
        OperandBase* APIConv2d(OperandBase*, OperandBase*, Conv2dOptions const* options);
        OperandBase* APIGemm(OperandBase*, OperandBase*, GemmOptions const* options);
        OperandBase* APILeakyRelu(OperandBase*, LeakyReluOptions const* options);
        FusionOperatorBase* APILeakyReluOperator(LeakyReluOptions const* options);
        OperandBase* APIMatmul(OperandBase* a, OperandBase* b);
        OperandBase* APIAveragePool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* APIMaxPool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* APIPad(OperandBase*, uint32_t const*, size_t, PadOptions const* options);
        OperandBase* APIReshape(OperandBase*, int32_t const*, size_t);
        OperandBase* APISigmoid(OperandBase*);
        FusionOperatorBase* APISigmoidOperator();
        OperandBase* APIRelu(OperandBase* x);
        FusionOperatorBase* APIReluOperator();
        OperandBase* APIReduceArgMax(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceArgMin(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceL1(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceL2(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceMax(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceMean(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceMin(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceProduct(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceSum(OperandBase*, ReduceOptions const* options);
        OperandBase* APIResample2d(OperandBase*, Resample2dOptions const* options);
        OperandBase* APISoftmax(OperandBase*);
        OperandBase* APITranspose(OperandBase*, TransposeOptions const* options);
        NamedOperandsBase* APICreateNamedOperands();
        GraphBase* APIBuild(NamedOperandsBase const* namedOperands);

      protected:
        GraphBuilderBase(DeviceBase* context);
        GraphBuilderBase(DeviceBase* device, ObjectBase::ErrorTag tag);
        virtual ~GraphBuilderBase() = default;

        // Topological sort of nodes needed to compute rootNodes
        std::vector<const OperatorBase*> TopologicalSort(
            std::vector<const OperandBase*>& rootNodes);

        virtual bool InitializeImpl();
        virtual GraphBase* CreateGraphImpl();
    };

}  // namespace dawn::native

#endif  // WEBNN_NATIVE_MODEL_BUILDER_H_
