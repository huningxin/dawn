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

#ifndef WEBNN_NATIVE_OPS_CONV2D_H_
#define WEBNN_NATIVE_OPS_CONV2D_H_

#include "dawn_native/FusionOperator.h"
#include "dawn_native/Graph.h"
#include "dawn_native/Operand.h"

namespace dawn::native { namespace op {

    void ComputeImplicitPaddingForAutoPad(wgpu::AutoPad autoPad,
                                          int32_t dilation,
                                          int32_t inputSize,
                                          int32_t filterSize,
                                          int32_t stride,
                                          int32_t& paddingBegin,
                                          int32_t& paddingEnd);

    class Conv2d final : public OperatorBase {
      public:
        Conv2d(GraphBuilderBase* builder,
               OperandBase* input,
               OperandBase* filter,
               Conv2dOptions const* options);
        ~Conv2d() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override;
        MaybeError ValidateAndInferOutputInfo() override;
        Conv2dOptions const* GetOptions() const;

      private:
        MaybeError CalculateShape();
        Conv2dOptions mOptions;
        std::vector<int32_t> mPadding;
        std::vector<int32_t> mStride;
        std::vector<int32_t> mDilations;
        Ref<FusionOperatorBase> mActivation;
    };

}}  // namespace dawn::native::op

#endif  // WEBNN_NATIVE_OPS_CONV2D_H_
