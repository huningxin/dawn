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

#include "dawn_native/ops/Pad.h"

#include "dawn_native/Error.h"

namespace dawn::native { namespace op {

    Pad::Pad(GraphBuilderBase* builder,
             OperandBase* input,
             uint32_t const* padding,
             size_t paddingCount,
             PadOptions const* options)
        : OperatorBase(builder, {input}) {
        mOptions.mode = options == nullptr ? wgpu::PaddingMode::Constant : options->mode;
        mOptions.value = options == nullptr ? 0 : options->value;
        mPadding.assign(padding, padding + paddingCount);
    }

    MaybeError Pad::CalculateShape() {
        auto inputShape = mInputs[0]->Shape();
        std::vector<int32_t> outputShape(inputShape.size());
        // For each dimension D of input, padding[D, 0] indicates how many values to add before the
        // content in that dimension, and padding[D, 1] indicates how many values to add after the
        // content in that dimension.
        for (size_t i = 0; i < inputShape.size(); ++i) {
            outputShape[i] = inputShape[i] + mPadding[2 * i] + mPadding[2 * i + 1];
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Pad::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        auto inputShape = mInputs[0]->Shape();
        if (inputShape.size() * 2 != mPadding.size()) {
            return DAWN_VALIDATION_ERROR(
                "The padding tensor should has shape [n, 2] where n is the rank of the input "
                "tensor.");
        }

        return CalculateShape();
    }

}}  // namespace dawn_native::op