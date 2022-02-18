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

#include "dawn_native/Graph.h"

#include <string>

#include "common/Assert.h"
#include "common/Log.h"
#include "common/RefCounted.h"
#include "dawn_native/Device.h"
#include "dawn_native/NamedResources.h"

namespace dawn::native {

    GraphBase::GraphBase(DeviceBase* device) : ObjectBase(device) {
    }

    MaybeError GraphBase::AddConstant(const op::Constant* constant) {
        return DAWN_UNIMPLEMENTED_ERROR("AddConstant");
    }

    MaybeError GraphBase::AddInput(const op::Input* input) {
        return DAWN_UNIMPLEMENTED_ERROR("AddInput");
    }

    MaybeError GraphBase::AddOutput(const std::string& name, const OperandBase* output) {
        return DAWN_UNIMPLEMENTED_ERROR("AddOutput");
    }

    MaybeError GraphBase::AddBinary(const op::Binary* binary) {
        return DAWN_UNIMPLEMENTED_ERROR("AddBinary");
    }

    MaybeError GraphBase::AddClamp(const op::Clamp* clamp) {
        return DAWN_UNIMPLEMENTED_ERROR("AddClamp");
    }

    MaybeError GraphBase::AddConv2d(const op::Conv2d* conv2d) {
        return DAWN_UNIMPLEMENTED_ERROR("AddConv2d");
    }

    MaybeError GraphBase::AddGemm(const op::Gemm* gemm) {
        return DAWN_UNIMPLEMENTED_ERROR("AddGemm");
    }

    MaybeError GraphBase::AddPool2d(const op::Pool2d* pool2d) {
        return DAWN_UNIMPLEMENTED_ERROR("AddPool2d");
    }

    MaybeError GraphBase::AddReshape(const op::Reshape* reshape) {
        return DAWN_UNIMPLEMENTED_ERROR("AddReshape");
    }

    MaybeError GraphBase::AddUnary(const op::Unary* unary) {
        return DAWN_UNIMPLEMENTED_ERROR("AddUnary");
    }
    
    MaybeError GraphBase::Finish() {
        return DAWN_UNIMPLEMENTED_ERROR("Finish");
    }

    MaybeError GraphBase::Compile() {
        return CompileImpl();
    }

    MaybeError GraphBase::CompileImpl() {
        return DAWN_UNIMPLEMENTED_ERROR("CompileImpl");
    }

    void GraphBase::ComputeImpl(NamedResourcesBase* inputs,
                                NamedResourcesBase* outputs) {
        dawn::ErrorLog() << "Unimplemented: GraphBase::ComputeImpl";
    }

    void GraphBase::APICompute(NamedResourcesBase* inputs, NamedResourcesBase* outputs) {
        DAWN_ASSERT(inputs != nullptr && outputs != nullptr);
        ComputeImpl(inputs, outputs);
    }

    
    NamedResourcesBase* GraphBase::APICreateNamedResources() {
        return new NamedResourcesBase();
    }

}  // namespace webnn_native
