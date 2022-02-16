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

#include "dawn_native/dml/GraphDML.h"

#include <algorithm>

#include "common/Assert.h"
#include "common/Log.h"
#include "dawn_native/ErrorData.h"
#include "dawn_native/NamedResources.h"
#include "dawn_native/d3d12/DeviceD3D12.h"
#include "dawn_native/dml/deps/src/precomp.h"

namespace dawn::native { namespace dml {

    namespace {
        enum TransposeType { NhwcToNchw, NchwToNhwc };

        bool CheckShape(const ::dml::Expression& expression,
                        const OperatorBase* operatorBase,
                        size_t index = 0) {
            DAWN_ASSERT(index < operatorBase->Outputs().size());
            auto expectedShape = operatorBase->Outputs()[index]->Shape();
            ::dml::TensorDimensions dmlShape = expression.GetOutputDesc().sizes;
            if (expectedShape.size() != dmlShape.size()) {
                dawn::ErrorLog() << "The size of output shape is expected as "
                                 << expectedShape.size() << ", but got " << dmlShape.size();
                return false;
            }
            for (size_t i = 0; i < dmlShape.size(); ++i) {
                if (expectedShape[i] < 0 || static_cast<size_t>(expectedShape[i]) != dmlShape[i]) {
                    dawn::ErrorLog() << "The output shape at index " << i << " is expected as "
                                     << expectedShape[i] << ", but got " << dmlShape[i];
                    return false;
                }
            }
            return true;
        }

        bool GetDmlTensorDataType(wgpu::OperandType operandType,
                                  DML_TENSOR_DATA_TYPE& dmlTensorDataType) {
            if (operandType == wgpu::OperandType::Float32) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
            } else if (operandType == wgpu::OperandType::Float16) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_FLOAT16;
            } else if (operandType == wgpu::OperandType::Int32) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_INT32;
            } else if (operandType == wgpu::OperandType::Uint32) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_UINT32;
            } else {
                return false;
            }
            return true;
        }

        bool GetDmlTensorDimensions(int32_t const* dimensions,
                                    uint32_t dimensionsCount,
                                    ::dml::TensorDimensions& dmlTensorDimensions) {
            if (dimensionsCount > DML_TENSOR_DIMENSION_COUNT_MAX) {
                dawn::ErrorLog() << "Tensor dimension count " << dimensionsCount
                                 << " is greater than DML_TENSOR_DIMENSION_COUNT_MAX "
                                 << DML_TENSOR_DIMENSION_COUNT_MAX;
                return false;
            }
            // for scale
            if (dimensionsCount == 0) {
                dmlTensorDimensions.resize(1);
                dmlTensorDimensions[0] = 1;
            } else {
                dmlTensorDimensions.resize(dimensionsCount);
                for (uint32_t i = 0; i < dimensionsCount; ++i) {
                    int32_t d = dimensions[i];
                    if (d < 0) {
                        dawn::ErrorLog() << "DML doesn't support the negative dimension value";
                        return false;
                    }
                    dmlTensorDimensions[i] = d;
                }
            }
            return true;
        }

        ::dml::TensorDimensions ExpandDimensions(const ::dml::TensorDimensions& dims, size_t rank) {
            DAWN_ASSERT(rank >= dims.size());
            ::dml::TensorDimensions newDims(rank, 1);
            for (size_t i = 0; i < dims.size(); ++i) {
                newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
            }
            return newDims;
        }

        ::dml::TensorDimensions ShrinkDimensions(const ::dml::TensorDimensions& dims, size_t rank) {
            DAWN_ASSERT(rank <= dims.size());
            ::dml::TensorDimensions newDims(rank);
            for (size_t i = 0; i < rank; ++i) {
                newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
            }
            return newDims;
        }

        // Strides are used to express broadcasting (by specifying a stride of 0) as well as
        // padding. If Strides is not specified, each dimension in the tensor is considered to
        // be contiguously packed, with no additional padding. The calculated strides refer to
        // https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides
        ::dml::TensorDimensions CalculateBroadcastStrides(::dml::TensorDimensions dims,
                                                          std::vector<bool> broadcast = {}) {
            size_t rank = dims.size();
            if (broadcast.empty()) {
                broadcast.resize(rank, false);
            }
            for (size_t i = 0; i < rank; ++i) {
                if (broadcast[i]) {
                    dims[i] = 1;
                }
            }
            ::dml::TensorDimensions strides(rank);
            strides[rank - 1] = broadcast[rank - 1] ? 0 : 1;
            size_t elements = 1;
            for (size_t i = 1; i < rank; i++) {
                size_t j = dims.size() - i - 1;
                elements *= dims[j + 1];
                strides[j] = broadcast[j] ? 0 : elements;
            }
            return strides;
        }

        bool BroadcastDimensions(const ::dml::TensorDimensions& aDims,
                                 const ::dml::TensorDimensions& bDims,
                                 bool& aBroadcasted,
                                 ::dml::TensorDimensions& aNewDims,
                                 ::dml::TensorDimensions& aNewStrides,
                                 bool& bBroadcasted,
                                 ::dml::TensorDimensions& bNewDims,
                                 ::dml::TensorDimensions& bNewStrides,
                                 size_t skipAxis = 0) {
            auto aRank = aDims.size();
            auto bRank = bDims.size();
            auto newRank = std::max(aRank, bRank);
            aNewDims.resize(newRank);
            aNewStrides.resize(newRank);
            std::vector<bool> aBroadcast(newRank, false);
            bNewDims.resize(newRank);
            bNewStrides.resize(newRank);
            std::vector<bool> bBroadcast(newRank, false);
            if (newRank > aRank) {
                aNewDims = ExpandDimensions(aDims, newRank);
                aBroadcasted = true;
            } else {
                aNewDims = aDims;
            }
            if (newRank > bRank) {
                bNewDims = ExpandDimensions(bDims, newRank);
                bBroadcasted = true;
            } else {
                bNewDims = bDims;
            }
            for (size_t i = 0; i < newRank - skipAxis; i++) {
                if (aNewDims[i] == 1 && bNewDims[i] != 1) {
                    aNewDims[i] = bNewDims[i];
                    aBroadcast[i] = true;
                    aBroadcasted = true;
                } else if (bNewDims[i] == 1 && aNewDims[i] != 1) {
                    bNewDims[i] = aNewDims[i];
                    bBroadcast[i] = true;
                    bBroadcasted = true;
                } else if (aNewDims[i] != bNewDims[i]) {
                    return false;
                }
            }
            aNewStrides = CalculateBroadcastStrides(aNewDims, aBroadcast);
            bNewStrides = CalculateBroadcastStrides(bNewDims, bBroadcast);
            return true;
        }

        ::dml::TensorDimensions CalculateInputLayoutStrides(TransposeType transposeType,
                                                            ::dml::TensorDimensions sizes) {
            uint32_t nStride = 0, cStride = 0, hStride = 0, wStride = 0;
            switch (transposeType) {
                case NhwcToNchw:
                    nStride = sizes[1] * sizes[2] * sizes[3];
                    hStride = sizes[2] * sizes[3];
                    wStride = sizes[3];
                    cStride = 1;
                    return {nStride, cStride, hStride, wStride};
                case NchwToNhwc:
                    nStride = sizes[1] * sizes[2] * sizes[3];
                    cStride = sizes[2] * sizes[3];
                    hStride = sizes[3];
                    wStride = 1;
                    return {nStride, hStride, wStride, cStride};
                default:
                    DAWN_ASSERT(0);
                    break;
            }
        }

        ::dml::Expression ReinterpretInputLayout(TransposeType transposeType,
                                                 ::dml::Expression input) {
            ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
            ::dml::TensorDimensions newInputDims;
            newInputDims.resize(4);
            switch (transposeType) {
                case NhwcToNchw:
                    newInputDims[0] = inputDims[0];
                    newInputDims[1] = inputDims[3];
                    newInputDims[2] = inputDims[1];
                    newInputDims[3] = inputDims[2];
                    input = ::dml::Reinterpret(input, newInputDims,
                                               CalculateInputLayoutStrides(NhwcToNchw, inputDims));
                    break;
                case NchwToNhwc:
                    newInputDims.resize(4);
                    newInputDims[0] = inputDims[0];
                    newInputDims[1] = inputDims[2];
                    newInputDims[2] = inputDims[3];
                    newInputDims[3] = inputDims[1];
                    input = ::dml::Reinterpret(input, newInputDims,
                                               CalculateInputLayoutStrides(NchwToNhwc, inputDims));
                    break;
                default:
                    DAWN_ASSERT(0);
                    break;
            }
            return input;
        }

        ::dml::TensorDimensions CalculateFilterLayoutStrides(wgpu::FilterOperandLayout filterLayout,
                                                             ::dml::TensorDimensions sizes) {
            uint32_t hStride = 0, wStride = 0, iStride = 0, oStride = 0;
            switch (filterLayout) {
                case wgpu::FilterOperandLayout::Hwio:
                    hStride = sizes[1] * sizes[2] * sizes[3];
                    wStride = sizes[2] * sizes[3];
                    iStride = sizes[3];
                    oStride = 1;
                    break;
                case wgpu::FilterOperandLayout::Ohwi:
                    oStride = sizes[1] * sizes[2] * sizes[3];
                    hStride = sizes[2] * sizes[3];
                    wStride = sizes[3];
                    iStride = 1;
                    break;
                case wgpu::FilterOperandLayout::Ihwo:
                    iStride = sizes[1] * sizes[2] * sizes[3];
                    hStride = sizes[2] * sizes[3];
                    wStride = sizes[3];
                    oStride = 1;
                    break;
                default:
                    DAWN_ASSERT(0);
                    break;
            }
            return {oStride, iStride, hStride, wStride};
        }

        ::dml::Expression ReinterpretFilterLayoutAsOihw(wgpu::FilterOperandLayout filterLayout,
                                                        ::dml::Expression filter) {
            ::dml::TensorDimensions filterDims = filter.GetOutputDesc().sizes;
            ::dml::TensorDimensions newFilterDims;
            newFilterDims.resize(4);
            switch (filterLayout) {
                case wgpu::FilterOperandLayout::Ohwi:
                    newFilterDims.resize(4);
                    newFilterDims[0] = filterDims[0];
                    newFilterDims[1] = filterDims[3];
                    newFilterDims[2] = filterDims[1];
                    newFilterDims[3] = filterDims[2];
                    filter = ::dml::Reinterpret(
                        filter, newFilterDims,
                        CalculateFilterLayoutStrides(wgpu::FilterOperandLayout::Ohwi, filterDims));
                    break;
                case wgpu::FilterOperandLayout::Hwio:
                    newFilterDims[0] = filterDims[3];
                    newFilterDims[1] = filterDims[2];
                    newFilterDims[2] = filterDims[0];
                    newFilterDims[3] = filterDims[1];
                    filter = ::dml::Reinterpret(
                        filter, newFilterDims,
                        CalculateFilterLayoutStrides(wgpu::FilterOperandLayout::Hwio, filterDims));
                    break;
                case wgpu::FilterOperandLayout::Ihwo:
                    newFilterDims[0] = filterDims[3];
                    newFilterDims[1] = filterDims[0];
                    newFilterDims[2] = filterDims[1];
                    newFilterDims[3] = filterDims[2];
                    filter = ::dml::Reinterpret(
                        filter, newFilterDims,
                        CalculateFilterLayoutStrides(wgpu::FilterOperandLayout::Ihwo, filterDims));
                    break;
                default:
                    DAWN_ASSERT(0);
                    break;
            }
            return filter;
        }

        ::dml::FusedActivation CreateFusedActivation(FusionOperatorBase* activation) {
            ::dml::FusedActivation dmlActivation = ::dml::FusedActivation::None();
            if (activation == nullptr) {
                return dmlActivation;
            }

            switch (activation->GetFusionType()) {
                case FusionType::Clamp:
                    return dmlActivation;
                case FusionType::Relu:
                    dmlActivation = ::dml::FusedActivation::Relu();
                    break;
                case FusionType::Sigmoid:
                    dmlActivation = ::dml::FusedActivation::Sigmoid();
                    break;
                case FusionType::LeakyRelu:
                    dmlActivation = ::dml::FusedActivation::LeakyRelu(
                        reinterpret_cast<op::FusionLeakyRelu*>(activation)->GetAlpha());
                    break;
                default:
                    DAWN_ASSERT(0);
            }
            return dmlActivation;
        }

        ::dml::Expression EmulateFusedActivation(FusionOperatorBase* activation,
                                                 ::dml::Expression& input) {
            if (activation == nullptr) {
                return input;
            }
            // HardSwish and Clamp are not supported for fusion, so we add them directly to emulate.
            // Currently we implement Relu6 operator by Clamp.
            auto type = activation->GetFusionType();
            if (type == FusionType::Clamp) {
                auto clamp = reinterpret_cast<const op::FusionClamp*>(activation);
                return ::dml::Clip(input, clamp->GetMinValue(), clamp->GetMaxValue());
            }
            return input;
        }

        std::string OpTypeToString(op::BinaryOpType type) {
            if (type == op::BinaryOpType::kAdd) {
                return "add";
            } else if (type == op::BinaryOpType::kMul) {
                return "mul";
            } else if (type == op::BinaryOpType::kSub) {
                return "sub";
            } else if (type == op::BinaryOpType::kDiv) {
                return "div";
            } else if (type == op::BinaryOpType::kMatMul) {
                return "matmul";
            }
            return std::to_string(type);
        }

        std::string OpTypeToString(op::UnaryOpType type) {
            if (type == op::UnaryOpType::kRelu) {
                return "relu";
            } else if (type == op::UnaryOpType::kSoftmax) {
                return "softmax";
            } else if (type == op::UnaryOpType::kSigmoid) {
                return "sigmoid";
            } else if (type == op::UnaryOpType::kTanh) {
                return "tanh";
            }
            return std::to_string(type);
        }

         template <typename T>
        std::vector<const uint32_t> ImplicitPadding(const T* options,
                                                    ::dml::Expression input,
                                                    std::vector<uint32_t> filterSize) {
            ::dml::Span<const uint32_t> strides(reinterpret_cast<const uint32_t*>(options->strides),
                                                options->stridesCount);
            ::dml::Span<const uint32_t> dilations(
                reinterpret_cast<const uint32_t*>(options->dilations), options->dilationsCount);
            ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
            // {paddingTop, paddingBottom, paddingLeft, paddingRight}
            int32_t paddingTop, paddingBottom, paddingLeft, paddingRight;
            dawn::native::op::ComputeImplicitPaddingForAutoPad(options->autoPad, dilations[0], inputDims[2],
                                                               filterSize[0], strides[0], paddingTop,
                                                               paddingBottom);
            dawn::native::op::ComputeImplicitPaddingForAutoPad(options->autoPad, dilations[1], inputDims[3],
                                                               filterSize[1], strides[1], paddingLeft,
                                                               paddingRight);
            return {static_cast<const uint32_t>(paddingTop),
                    static_cast<const uint32_t>(paddingBottom),
                    static_cast<const uint32_t>(paddingLeft),
                    static_cast<const uint32_t>(paddingRight)};
        }

        template <typename T>
        std::vector<const uint32_t> ImplicitPadding(const T* options,
                                                    ::dml::Expression input,
                                                    ::dml::Expression filter) {
            ::dml::TensorDimensions filterDims = filter.GetOutputDesc().sizes;
            return ImplicitPadding(options, input, {filterDims[2], filterDims[3]});
        }

        template <typename T>
        std::vector<const uint32_t> ExplicitPadding(const T* options) {
            uint32_t paddingTop = static_cast<uint32_t>(options->padding[0]);
            uint32_t paddingBottom = static_cast<uint32_t>(options->padding[1]);
            uint32_t paddingLeft = static_cast<uint32_t>(options->padding[2]);
            uint32_t paddingRight = static_cast<uint32_t>(options->padding[3]);

            return {static_cast<const uint32_t>(paddingTop),
                    static_cast<const uint32_t>(paddingBottom),
                    static_cast<const uint32_t>(paddingLeft),
                    static_cast<const uint32_t>(paddingRight)};
        }
    }  // namespace

    std::string DmlTensorDimensionsToString(const ::dml::TensorDimensions& dimensions) {
        std::string output = "[";
        for (size_t i = 0; i < dimensions.size(); ++i) {
            output.append(std::to_string(dimensions[i]));
            if (i != dimensions.size() - 1) {
                output.append(",");
            }
        }
        output.append("]");
        return output;
    }

    template <typename T>
    std::string DmlSpanToString(const ::dml::Span<T>& span) {
        std::string output = "[";
        for (size_t i = 0; i < span.size(); ++i) {
            output.append(std::to_string(span[i]));
            if (i != span.size() - 1) {
                output.append(",");
            }
        }
        output.append("]");
        return output;
    }

    std::string DmlTensorDataTypeToString(DML_TENSOR_DATA_TYPE type) {
        if (type == DML_TENSOR_DATA_TYPE_UNKNOWN) {
            return "UNKNOWN";
        } else if (type == DML_TENSOR_DATA_TYPE_FLOAT32) {
            return "FLOAT32";
        } else if (type == DML_TENSOR_DATA_TYPE_FLOAT16) {
            return "FLOAT16";
        } else if (type == DML_TENSOR_DATA_TYPE_UINT32) {
            return "UINT32";
        } else if (type == DML_TENSOR_DATA_TYPE_UINT16) {
            return "UINT16";
        } else if (type == DML_TENSOR_DATA_TYPE_UINT8) {
            return "UINT8";
        } else if (type == DML_TENSOR_DATA_TYPE_INT32) {
            return "INT32";
        } else if (type == DML_TENSOR_DATA_TYPE_INT16) {
            return "INT16";
        } else if (type == DML_TENSOR_DATA_TYPE_INT8) {
            return "INT8";
        } else if (type == DML_TENSOR_DATA_TYPE_FLOAT64) {
            return "FLOAT64";
        } else if (type == DML_TENSOR_DATA_TYPE_UINT64) {
            return "UINT64";
        } else if (type == DML_TENSOR_DATA_TYPE_INT64) {
            return "INT64";
        }
        return std::to_string(type);
    }

    Graph::Graph(DeviceBase* device) : GraphBase(device) {
        d3d12::Device* d3d12Device = reinterpret_cast<d3d12::Device*>(device);
        mDevice.reset(new ::pydml::Device(d3d12Device));
        mDevice->Init();
        mGraph.reset(new ::dml::Graph(mDevice->GetDevice()));
    }

    ::dml::Expression Graph::BindingConstant(DML_TENSOR_DATA_TYPE dmlTensorType,
                                             ::dml::TensorDimensions dmlTensorDims,
                                             BufferBase* buffer,
                                             size_t offset,
                                             size_t size) {
        ::dml::TensorDesc dmlTensorDesc(dmlTensorType,
                                        ::DML_TENSOR_FLAGS::DML_TENSOR_FLAG_OWNED_BY_DML,
                                        dmlTensorDims, ::dml::TensorPolicy::Default());
        ::dml::Expression dmlConstant =
            ::dml::InputTensor(*mGraph, mInputBindings.size(), dmlTensorDesc);
        std::unique_ptr<::pydml::Binding> binding(
            new ::pydml::Binding(dmlConstant, reinterpret_cast<d3d12::Buffer*>(buffer), size, offset));
        mInputBindings.push_back(std::move(binding));
        return dmlConstant;
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        const OperandDescriptor* desc = constant->GetOperandDescriptor();
        DML_TENSOR_DATA_TYPE dmlTensorType;
        if (!GetDmlTensorDataType(desc->type, dmlTensorType)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML tensor type.");
        }
        ::dml::TensorDimensions dmlTensorDims;
        if (!GetDmlTensorDimensions(desc->dimensions, desc->dimensionsCount, dmlTensorDims)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML tensor dimensions.");
        }

        auto dmlConstant = BindingConstant(dmlTensorType, dmlTensorDims, constant->GetBuffer(),
                                           constant->GetOffset(), constant->GetSize());
        mExpression.insert(std::make_pair(constant->PrimaryOutput(), dmlConstant));
        mConstantSet.insert(constant->PrimaryOutput());
        DAWN_ASSERT(CheckShape(dmlConstant, constant));
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        const OperandDescriptor* desc = input->GetOperandDescriptor();
        DML_TENSOR_DATA_TYPE dmlTensorType;
        if (!GetDmlTensorDataType(desc->type, dmlTensorType)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML tensor type.");
        }
        ::dml::TensorDimensions dmlTensorDims;
        if (!GetDmlTensorDimensions(desc->dimensions, desc->dimensionsCount, dmlTensorDims)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML tensor dimensions.");
        }
        ::dml::TensorDesc dmlTensorDesc(dmlTensorType, dmlTensorDims,
                                        ::dml::TensorPolicy::Default());
        ::dml::Expression dmlInput = ::dml::InputTensor(*mGraph, mInputBindings.size(), dmlTensorDesc);
        mExpression.insert(std::make_pair(input->PrimaryOutput(), dmlInput));
        std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(dmlInput, nullptr, 0));
        mInputBindings.push_back(std::move(binding));
        mInputs.insert(std::make_pair(input->GetName(), mInputBindings.back().get()));
        DAWN_ASSERT(CheckShape(dmlInput, input));
        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        DAWN_ASSERT(mExpression.find(output) != mExpression.end());
        ::dml::Expression dmlOutput = mExpression.at(output);
        mOutputExpressions.push_back(dmlOutput);
        std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(dmlOutput, nullptr, 0));
        mOutputBindings.push_back(std::move(binding));
        mOutputs.insert(std::make_pair(name, mOutputBindings.back().get()));
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        DAWN_ASSERT(binary->Inputs().size() == 2);
        DAWN_ASSERT(mExpression.find(binary->Inputs()[0].Get()) != mExpression.end());
        ::dml::Expression a = mExpression.at(binary->Inputs()[0].Get());
        DAWN_ASSERT(mExpression.find(binary->Inputs()[1].Get()) != mExpression.end());
        ::dml::Expression b = mExpression.at(binary->Inputs()[1].Get());
        ::dml::Expression c;
        ::dml::TensorDimensions aDims = a.GetOutputDesc().sizes;
        const size_t aRank = aDims.size();
        ::dml::TensorDimensions bDims = b.GetOutputDesc().sizes;
        const size_t bRank = bDims.size();
        ::dml::TensorDimensions aNewDims, bNewDims;
        ::dml::TensorDimensions aNewStrides, bNewStrides;
        bool aDimsChanged = false, bDimsChanged = false;
        size_t cRank = 0;
        bool needBroadcast = false;
        size_t broadcastSkipAxis = 0;

        if (binary->GetType() == op::BinaryOpType::kMatMul) {
            // DML GEMM requires inputs are either 4D or 5D. We use 4D.
            if (aRank > 4 || bRank > 4) {
                return DAWN_INTERNAL_ERROR("The size of input dimensions is greater than 4.");
            }

            if (aRank == 1 && bRank == 1) {
                // If both a and b are 1-D, the operation is a vector dot-product,
                // which produces a scalar output.
                cRank = 1;
            } else {
                // The output is a N-D tensor whose rank is the maximum rank of the
                // input tensors.
                cRank = std::max(aRank, bRank);
            }

            if (aRank < 4) {
                aDims = ExpandDimensions(aDims, 4);
                aDimsChanged = true;
                aNewDims = aDims;
                aNewStrides = CalculateBroadcastStrides(aNewDims);
            }

            if (bRank < 4) {
                if (bRank == 1) {
                    // If b is 1-D, it is converted to a 2-D tensor by by appending a 1 to
                    // its dimensions.
                    bDims.push_back(1);
                }
                bDims = ExpandDimensions(bDims, 4);
                bDimsChanged = true;
                bNewDims = bDims;
                bNewStrides = CalculateBroadcastStrides(bNewDims);
            }

            if (aRank > 2 || bRank > 2) {
                // If either a or b is N-D, N > 2, it is treated as a stack of matrices
                // with dimensions corresponding to the last two indices. The matrix
                // multiplication will be broadcasted accordingly by following
                // [numpy-broadcasting-rule].
                needBroadcast = true;
                broadcastSkipAxis = 2;
            }
        } else {
            // The element-wise binary operation will be broadcasted according to
            // [numpy-broadcasting-rule].
            needBroadcast = true;
            broadcastSkipAxis = 0;
        }

        if (needBroadcast) {
            if (!BroadcastDimensions(aDims, bDims, aDimsChanged, aNewDims, aNewStrides,
                                     bDimsChanged, bNewDims, bNewStrides, broadcastSkipAxis)) {
                return DAWN_INTERNAL_ERROR("Failed to broadcast a and b.");
            }
        }

        if (aDimsChanged) {
            a = ::dml::Reinterpret(a, aNewDims, aNewStrides);
        }
        if (bDimsChanged) {
            b = ::dml::Reinterpret(b, bNewDims, bNewStrides);
        }

        if (binary->GetType() == op::BinaryOpType::kMatMul) {
            c = ::dml::Gemm(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kAdd) {
            c = ::dml::Add(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kDiv) {
            c = ::dml::Divide(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kMul) {
            c = ::dml::Multiply(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kSub) {
            c = ::dml::Subtract(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kMax) {
            c = ::dml::Max(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kMin) {
            c = ::dml::Min(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kPower) {
            c = ::dml::Pow(a, b);
        } else {
            std::string errorMessage = std::string(" Binary op ") +
                                       OpTypeToString(binary->GetType()) +
                                       std::string(" is not implemented.");
            return DAWN_UNIMPLEMENTED_ERROR(errorMessage);
        }

        // Reshape back according to c rank if needed.
        ::dml::TensorDimensions cDims = c.GetOutputDesc().sizes;
        if (cRank != 0 && cRank < cDims.size()) {
            ::dml::TensorDimensions cNewDims = ShrinkDimensions(cDims, cRank);
            c = ::dml::Reinterpret(c, cNewDims, ::dml::NullOpt);
        }
        mExpression.insert(std::make_pair(binary->PrimaryOutput(), c));
        DAWN_ASSERT(CheckShape(c, binary));
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        auto inputsOperand = clamp->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 1);
        ::dml::Expression input = mExpression.at(inputsOperand[0].Get());
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        if (inputDims.size() > DML_TENSOR_DIMENSION_COUNT_MAX1) {
            return DAWN_INTERNAL_ERROR("The size of input dimensions is greater than max");
        }
        auto output = ::dml::Clip(input, clamp->GetMinValue(), clamp->GetMaxValue());
        mExpression.insert(std::make_pair(clamp->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, clamp));
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        auto inputsOperand = conv2d->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 2 || inputsOperand.size() == 3);
        DAWN_ASSERT(mExpression.find(inputsOperand[0].Get()) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputsOperand[0].Get());
        DAWN_ASSERT(mExpression.find(inputsOperand[1].Get()) != mExpression.end());
        ::dml::Expression filter = mExpression.at(inputsOperand[1].Get());
        const Conv2dOptions* options = conv2d->GetOptions();

        if (options->inputLayout == wgpu::InputOperandLayout::Nhwc) {
            input = ReinterpretInputLayout(NhwcToNchw, input);
        }

        if (options->filterLayout != wgpu::FilterOperandLayout::Oihw) {
            filter = ReinterpretFilterLayoutAsOihw(options->filterLayout, filter);
        }

        // FIXME(nhu): strides, dilations, padding should be uint32_t
        // need to fix the spec.
        ::dml::Span<const uint32_t> strides(reinterpret_cast<const uint32_t*>(options->strides),
                                            options->stridesCount);
        ::dml::Span<const uint32_t> dilations(reinterpret_cast<const uint32_t*>(options->dilations),
                                              options->dilationsCount);

        auto padding = options->autoPad == wgpu::AutoPad::Explicit
                           ? ExplicitPadding<Conv2dOptions>(options)
                           : ImplicitPadding<Conv2dOptions>(options, input, filter);
        // dml::Span just holds the refernces, need a variable to hold the memory.
        std::vector<const uint32_t> startPaddingVector = {padding[0], padding[2]};
        ::dml::Span<const uint32_t> startPadding(startPaddingVector);
        std::vector<const uint32_t> endPaddingVector = {padding[1], padding[3]};
        ::dml::Span<const uint32_t> endPadding(endPaddingVector);

        ::dml::Optional<::dml::Expression> bias = ::dml::NullOpt;
        if (options->bias != nullptr) {
            DAWN_ASSERT(mExpression.find(inputsOperand[2].Get()) != mExpression.end());
            bias = mExpression.at(inputsOperand[2].Get());
            ::dml::TensorDimensions biasDims = bias->GetOutputDesc().sizes;
            if (biasDims[0] != filter.GetOutputDesc().sizes[0] || biasDims.size() != 1) {
                return DAWN_INTERNAL_ERROR(
                    "The bias should be 1-D tensor with the shape of [output_channels].");
            }
            // Reshape bias from 1-D to 4-D for NCHW layout.
            ::dml::TensorDimensions expandDimens = {1, biasDims[0], 1, 1};
            bias = ::dml::Reinterpret(*bias, expandDimens, ::dml::NullOpt);
        }
        ::dml::Expression output = ::dml::Convolution(
            input, filter, bias, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
            DML_CONVOLUTION_DIRECTION_FORWARD, strides, dilations, startPadding, endPadding,
            // outPadding
            {},
            // groupCount
            options->groups, CreateFusedActivation(options->activation));
        if (options->inputLayout == wgpu::InputOperandLayout::Nhwc) {
            output = ::dml::Identity(ReinterpretInputLayout(NchwToNhwc, output));
        }
        output = EmulateFusedActivation(options->activation, output);
        mExpression.insert(std::make_pair(conv2d->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, conv2d));
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        DAWN_ASSERT(reshape->Inputs().size() == 1);
        const OperandBase* inputOperand = reshape->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        auto newShape = reshape->GetNewShape();
        if (newShape.size() > DML_TENSOR_DIMENSION_COUNT_MAX) {
            return DAWN_INTERNAL_ERROR("The size of new shape is not supported by DML.");
        }
        ::dml::TensorDimensions newSizes(newShape.size());
        uint32_t outputElementCount = 1;
        int32_t inferAxis = -1;

        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        uint32_t inputElementCount =
            std::accumulate(inputDims.begin(), inputDims.end(), 1, std::multiplies<uint32_t>());

        for (size_t i = 0; i < newShape.size(); ++i) {
            if (newShape[i] == -1) {
                if (inferAxis != -1) {
                    return DAWN_VALIDATION_ERROR("New shape should contain only one -1 value.");
                } else {
                    inferAxis = i;
                }
            } else if (newShape[i] <= 0) {
                return DAWN_VALIDATION_ERROR("Argument new shape is invalid");
            } else {
                newSizes[i] = newShape[i];
                outputElementCount *= newSizes[i];
            }
        }

        if (inferAxis != -1) {
            newSizes[inferAxis] = inputElementCount / outputElementCount;
        }

        ::dml::Expression output = ::dml::Reinterpret(input, newSizes, ::dml::NullOpt);
        mExpression.insert(std::make_pair(reshape->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, reshape));
        return {};
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        DAWN_ASSERT(unary->Inputs().size() == 1);
        const OperandBase* inputOperand = unary->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        if (inputDims.size() > DML_TENSOR_DIMENSION_COUNT_MAX1) {
            return DAWN_INTERNAL_ERROR("The size of input dimensions isn't supported.");
        }

        ::dml::Expression output;
        switch (unary->GetType()) {
            case op::UnaryOpType::kAbs:
                output = ::dml::Abs(input);
                break;
            case op::UnaryOpType::kCeil:
                output = ::dml::Ceil(input);
                break;
            case op::UnaryOpType::kCos:
                output = ::dml::Cos(input);
                break;
            case op::UnaryOpType::kExp:
                output = ::dml::Exp(input);
                break;
            case op::UnaryOpType::kFloor:
                output = ::dml::Floor(input);
                break;
            case op::UnaryOpType::kLog:
                output = ::dml::Log(input);
                break;
            case op::UnaryOpType::kLeakyRelu:
                output = ::dml::ActivationLeakyRelu(
                    input, reinterpret_cast<const op::LeakyRelu*>(unary)->GetAlpha());
                break;
            case op::UnaryOpType::kRelu:
                output = ::dml::ActivationRelu(input);
                break;
            case op::UnaryOpType::kSigmoid:
                output = ::dml::ActivationSigmoid(input);
                break;
            case op::UnaryOpType::kSin:
                output = ::dml::Sin(input);
                break;
            case op::UnaryOpType::kSoftmax:
                output = ::dml::ActivationSoftmax(input);
                break;
            case op::UnaryOpType::kTan:
                output = ::dml::Tan(input);
                break;
            case op::UnaryOpType::kTanh:
                output = ::dml::ActivationTanh(input);
                break;
            default:
                return DAWN_UNIMPLEMENTED_ERROR(" Unary op " + OpTypeToString(unary->GetType()) +
                                                " is not implemented.");
        }
        mExpression.insert(std::make_pair(unary->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, unary));
        return {};
    }

    MaybeError Graph::Finish() {
        if (mInputs.empty()) {
            return DAWN_VALIDATION_ERROR("Model inputs must be set.");
        }
        if (mOutputExpressions.size() == 1) {
            auto outputExp = mOutputExpressions[0];
            if (outputExp.Impl()->GetNode().type == ::dml::detail::NodeType::Reinterpret) {
                // Deal with a graph with single reshape node.
                // https://github.com/microsoft/DirectML/issues/71
                mOutputExpressions[0] = ::dml::ActivationIdentity(outputExp);
            }
        }

        return {};
    }

    MaybeError Graph::CompileImpl() {
        // TODO(nhu): investigate other execution flag,
        // e.g. DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
        mCompiledModel.reset(new pydml::CompiledModel(*(mGraph), DML_EXECUTION_FLAG_NONE, mOutputExpressions));

        std::vector<pydml::Binding*> inputBindings;
        for (auto& binding : mInputBindings) {
            inputBindings.push_back(binding.get());
        }
        std::lock_guard<std::mutex> lock(mMutex);
        if (FAILED(mDevice->InitializeOperator(mCompiledModel->op.Get(), inputBindings))) {
            return DAWN_INTERNAL_ERROR("Failed to compile graph.");
        }
        return {};
    }

    void Graph::ComputeImpl(NamedResourcesBase* inputs, NamedResourcesBase* outputs) {
        auto namedInputs = inputs->GetResources();
        for (auto& input : mInputs) {
            // All the inputs must be set.
            if (namedInputs.find(input.first) == namedInputs.end()) {
                dawn::ErrorLog() << "The input must be set.";
                return;
            }

            ::pydml::Binding* binding = input.second;
            auto& bufferView = namedInputs[input.first];
            binding->data.buffer = reinterpret_cast<d3d12::Buffer*>(bufferView.resource);
            binding->data.offset = bufferView.offset;
            binding->data.size = bufferView.size != 0 ? bufferView.size : bufferView.resource->GetSize();
        }
        std::vector<pydml::Binding*> inputBindings;
        for (auto& binding : mInputBindings) {
            inputBindings.push_back(binding.get());
        }
        std::vector<pydml::Binding*> outputBindings;
        auto namedOutputs = outputs->GetResources();
        for (auto& output : mOutputs) {
            ::pydml::Binding* binding = output.second;
            auto& bufferView = namedOutputs[output.first];
            binding->data.buffer = reinterpret_cast<d3d12::Buffer*>(bufferView.resource);
            binding->data.offset = bufferView.offset;
            binding->data.size = bufferView.size != 0 ? bufferView.size : bufferView.resource->GetSize();
            outputBindings.push_back(binding);
        }
        std::lock_guard<std::mutex> lock(mMutex);
        if (FAILED(mDevice->DispatchOperator(mCompiledModel->op.Get(), inputBindings, outputBindings))) {
            dawn::ErrorLog() << "Failed to dispatch operator.";
        }
    }

}}  // namespace dawn::native::dml
