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

#include "dawn/native/ops/Conv2d.h"

#include "dawn/native/Error.h"
#include "dawn/native/Operator.h"

namespace dawn::native { namespace op {

    void ComputeImplicitPaddingForAutoPad(wgpu::AutoPad autoPad,
                                          int32_t dilation,
                                          int32_t inputSize,
                                          int32_t filterSize,
                                          int32_t stride,
                                          int32_t& paddingBegin,
                                          int32_t& paddingEnd) {
        int32_t outSize = (inputSize + stride - 1) / stride;
        int32_t dilatedFilter = (filterSize - 1) * dilation + 1;
        int32_t neededInput = (outSize - 1) * stride + dilatedFilter;
        int32_t totalPadding = neededInput > inputSize ? neededInput - inputSize : 0;
        switch (autoPad) {
            case wgpu::AutoPad::SameUpper:
                paddingBegin = totalPadding / 2;
                paddingEnd = (totalPadding + 1) / 2;
                break;
            case wgpu::AutoPad::SameLower:
                paddingBegin = (totalPadding + 1) / 2;
                paddingEnd = totalPadding / 2;
                break;
            default:
                DAWN_UNREACHABLE();
        }
    }

    Conv2d::Conv2d(GraphBuilderBase* builder,
                   OperandBase* input,
                   OperandBase* filter,
                   Conv2dOptions const* options)
        : OperatorBase(builder, {input, filter}) {
        if (options != nullptr && options->bias != nullptr) {
            mInputs.push_back(options->bias);
        }
        if (options == nullptr || options->paddingCount == 0 || options->padding == nullptr) {
            mPadding = std::vector<int32_t>(4, 0);
        } else {
            mPadding.assign(options->padding, options->padding + options->paddingCount);
        }
        mOptions.padding = mPadding.data();
        mOptions.paddingCount = mPadding.size();

        if (options == nullptr || options->stridesCount == 0 || options->strides == nullptr) {
            mStride = std::vector<int32_t>(2, 1);
        } else {
            mStride.assign(options->strides, options->strides + options->stridesCount);
        }
        mOptions.strides = mStride.data();
        mOptions.stridesCount = mStride.size();

        if (options == nullptr || options->dilationsCount ==0 || options->dilations == nullptr) {
            mDilations = std::vector<int32_t>(2, 1);
        } else {
            mDilations.assign(options->dilations, options->dilations + options->dilationsCount);
        }
        mOptions.dilations = mDilations.data();
        mOptions.dilationsCount = mDilations.size();

        if (options != nullptr) {
            mOptions.groups = options->groups;
            mOptions.inputLayout = options->inputLayout;
            mOptions.filterLayout = options->filterLayout;
            mOptions.autoPad = options->autoPad;
            mOptions.bias = options->bias;
            mOptions.activation = options->activation;
        }
        mActivation = Ref<FusionOperatorBase>(mOptions.activation);
    }

    MaybeError Conv2d::AddToGraph(GraphBase* graph) const {
        return graph->AddConv2d(this);
    }

    Conv2dOptions const* Conv2d::GetOptions() const {
        return &mOptions;
    }

    MaybeError Conv2d::CalculateShape() {
        auto inputShape = mInputs[0]->Shape();
        auto filterShape = mInputs[1]->Shape();

        bool nchw = mOptions.inputLayout == wgpu::InputOperandLayout::Nchw;
        int32_t inputHeight = nchw ? inputShape[2] : inputShape[1];
        int32_t inputWidth = nchw ? inputShape[3] : inputShape[2];
        int32_t inputChannels = nchw ? inputShape[1] : inputShape[3];

        int32_t filterHeight = 0, filterWidth = 0, outputChannels = 0, filterDepthIn = 0;
        switch (mOptions.filterLayout) {
            case wgpu::FilterOperandLayout::Hwio:
                filterHeight = filterShape[0];
                filterWidth = filterShape[1];
                outputChannels = filterShape[3];
                filterDepthIn = filterShape[2];
                break;
            case wgpu::FilterOperandLayout::Ohwi:
                filterHeight = filterShape[1];
                filterWidth = filterShape[2];
                outputChannels = filterShape[0];
                filterDepthIn = filterShape[3];
                break;
            case wgpu::FilterOperandLayout::Ihwo:
                filterHeight = filterShape[1];
                filterWidth = filterShape[2];
                outputChannels = filterShape[3];
                filterDepthIn = filterShape[0];
                break;
            case wgpu::FilterOperandLayout::Oihw:
                filterHeight = filterShape[2];
                filterWidth = filterShape[3];
                outputChannels = filterShape[0];
                filterDepthIn = filterShape[1];
                break;
            default:
                return DAWN_VALIDATION_ERROR("The filter layout is unsupported");
        }

        if (filterDepthIn != inputChannels / mOptions.groups) {
            return DAWN_VALIDATION_ERROR(
                "The groups is invalid, it must evenly divides the input channels.");
        }

        int32_t paddingBeginningHeight = mPadding[0], paddingEndingHeight = mPadding[1],
                paddingBeginningWidth = mPadding[2], paddingEndingWidth = mPadding[3];
        if (mOptions.autoPad != wgpu::AutoPad::Explicit) {
            ComputeImplicitPaddingForAutoPad(mOptions.autoPad, mOptions.dilations[0],
                                             inputHeight, filterHeight, mOptions.strides[0],
                                             paddingBeginningHeight, paddingEndingHeight);
            ComputeImplicitPaddingForAutoPad(mOptions.autoPad, mOptions.dilations[1],
                                             inputWidth, filterWidth, mOptions.strides[1],
                                             paddingBeginningWidth, paddingEndingWidth);
        }

        int32_t outputHeight, outputWidth;
        auto dilatedFilterHeight = mDilations[0] * (filterHeight - 1) + 1;
        auto dilatedFilterWidth = mDilations[1] * (filterWidth - 1) + 1;
        outputHeight = 1 + (inputHeight - dilatedFilterHeight + paddingBeginningHeight +
                            paddingEndingHeight) /
                                mStride[0];
        outputWidth =
            1 + (inputWidth - dilatedFilterWidth + paddingBeginningWidth + paddingEndingWidth) /
                    mStride[1];

        std::vector<int32_t> outputShape;
        auto batches = inputShape[0];
        if (nchw) {
            outputShape = {batches, outputChannels, outputHeight, outputWidth};
        } else {
            outputShape = {batches, outputHeight, outputWidth, outputChannels};
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Conv2d::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        auto input = mInputs[0];
        auto filter = mInputs[1];
        if (input->Type() != filter->Type()) {
            return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
        }
        // The input 4-D tensor
        if (input->Shape().size() != 4) {
            return DAWN_VALIDATION_ERROR("Argument input is not a 4D tensor.");
        }
        // The filter 4-D tensor
        if (filter->Shape().size() != 4) {
            return DAWN_VALIDATION_ERROR("Argument filter is not a 4D tensor.");
        }
        // The bias is 1-D tensor.
        if (mOptions.bias != nullptr) {
            auto bias = mInputs[2];
            if (bias->Shape().size() != 1) {
                return DAWN_VALIDATION_ERROR("Argument bias is not a 1D tensor.");
            }
        }
        // padding: a sequence of long of length 4
        if (mOptions.paddingCount != 4) {
            return DAWN_VALIDATION_ERROR("PaddingCount is incorrect.");
        }
        // strides: a sequence of long of length 2
        if (mOptions.stridesCount != 2) {
            return DAWN_VALIDATION_ERROR("stridesCount is incorrect.");
        }
        // dilations: a sequence of long of length 2
        if (mOptions.dilationsCount != 2) {
            return DAWN_VALIDATION_ERROR("dilationsCount is incorrect.");
        }

        return CalculateShape();
    }

}}  // namespace dawn::native::op
