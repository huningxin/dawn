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

#include "common/RefCounted.h"
#include "dawn_native/Forward.h"
#include "dawn_native/Device.h"
#include "dawn_native/NamedOperands.h"
#include "dawn_native/ObjectBase.h"
#include "dawn_native/Operand.h"
#include "dawn_native/Operator.h"
#include "dawn_native/dawn_platform.h"

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
        OperandBase* APIClamp(OperandBase*, ClampOptions const* options);
        FusionOperatorBase* APIClampOperator(ClampOptions const* options);
        OperandBase* APIConv2d(OperandBase*, OperandBase*, Conv2dOptions const* options);
        OperandBase* APIGemm(OperandBase*, OperandBase*, GemmOptions const* options);
        OperandBase* APILeakyRelu(OperandBase*, LeakyReluOptions const* options);
        FusionOperatorBase* APILeakyReluOperator(LeakyReluOptions const* options);
        OperandBase* APIMatmul(OperandBase* a, OperandBase* b);
        OperandBase* APIAveragePool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* APIMaxPool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* APIReshape(OperandBase*, int32_t const*, size_t);
        OperandBase* APISigmoid(OperandBase*);
        FusionOperatorBase* APISigmoidOperator();
        OperandBase* APIRelu(OperandBase* x);
        FusionOperatorBase* APIReluOperator();
        OperandBase* APISoftmax(OperandBase*);
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
