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

namespace dawn_native { namespace dml {

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
#if defined(_DEBUG)
        mDevice.reset(new ::pydml::Device(d3d12Device->GetD3D12Device(), d3d12Device->GetCommandQueue(), true));
#else
        mDevice.reset(new ::pydml::Device(d3d12Device->GetD3D12Device(), d3d12Device->GetCommandQueue(), false));
#endif
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
            ::dml::InputTensor(*mGraph, mBindings.size(), dmlTensorDesc);
        ID3D12Resource* d3d12Resource = reinterpret_cast<d3d12::Buffer*>(buffer)->GetD3D12Resource();
        std::unique_ptr<::pydml::Binding> binding(
            new ::pydml::Binding(dmlConstant, d3d12Resource, offset, size));
        mBindings.push_back(std::move(binding));
        mConstants.push_back(mBindings.back().get());
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
        ::dml::Expression dmlInput = ::dml::InputTensor(*mGraph, mBindings.size(), dmlTensorDesc);
        mExpression.insert(std::make_pair(input->PrimaryOutput(), dmlInput));
        std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(dmlInput, nullptr, 0));
        mBindings.push_back(std::move(binding));
        mInputs.insert(std::make_pair(input->GetName(), mBindings.back().get()));
        DAWN_ASSERT(CheckShape(dmlInput, input));
        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        DAWN_ASSERT(mExpression.find(output) != mExpression.end());
        ::dml::Expression dmlOutput = mExpression.at(output);
        mOutputExpressions.push_back(dmlOutput);
        std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(dmlOutput, nullptr, 0));
        mBindings.push_back(std::move(binding));
        mOutputs.insert(std::make_pair(name, mBindings.back().get()));
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

        return {};
    }

    MaybeError Graph::CompileImpl() {
        // TODO(nhu): investigate other execution flag,
        // e.g. DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
        mCompiledModel.reset(new pydml::CompiledModel(*(mGraph), DML_EXECUTION_FLAG_NONE, mOutputExpressions));

        std::lock_guard<std::mutex> lock(mMutex);
        if (FAILED(mDevice->InitializeOperator(mCompiledModel->op.Get(), mConstants))) {
            return DAWN_INTERNAL_ERROR("Failed to compile graph.");
        }
        return {};
    }

    void Graph::ComputeImpl(NamedResourcesBase* inputs, NamedResourcesBase* outputs) {
        std::vector<pydml::Binding*> inputBindings;
        auto namedInputs = inputs->GetResources();
        for (auto& input : mInputs) {
            // All the inputs must be set.
            if (namedInputs.find(input.first) == namedInputs.end()) {
                dawn::ErrorLog() << "The input must be set.";
                return;
            }

            ::pydml::Binding* binding = input.second;
            auto& bufferView = namedInputs[input.first];
            binding->data.buffer = reinterpret_cast<d3d12::Buffer*>(bufferView.resource)->GetD3D12Resource();
            binding->data.offset = bufferView.offset;
            binding->data.size = bufferView.size;
            inputBindings.push_back(binding);
        }
        std::vector<pydml::Binding*> outputBindings;
        auto namedOutputs = outputs->GetResources();
        for (auto& output : mOutputs) {
            ::pydml::Binding* binding = output.second;
            auto& bufferView = namedOutputs[output.first];
            binding->data.buffer = reinterpret_cast<d3d12::Buffer*>(bufferView.resource)->GetD3D12Resource();
            binding->data.offset = bufferView.offset;
            binding->data.size = bufferView.size;
            outputBindings.push_back(binding);
        }
        std::lock_guard<std::mutex> lock(mMutex);
        if (FAILED(mDevice->DispatchOperator(mCompiledModel->op.Get(), inputBindings, outputBindings))) {
            dawn::ErrorLog() << "Failed to dispatch operator.";
        }
    }

}}  // namespace dawn_native::dml
