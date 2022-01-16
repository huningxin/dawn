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
        dawn::ErrorLog() << "Unimplemented: GraphBase::AddConstant";
        return {};
    }

    MaybeError GraphBase::AddInput(const op::Input* input) {
        dawn::ErrorLog() << "Unimplemented: GraphBase::AddInput";
        return {};
    }

    MaybeError GraphBase::AddOutput(const std::string& name, const OperandBase* output) {
        dawn::ErrorLog() << "Unimplemented: GraphBase::AddOutput";
        return {};
    }

    MaybeError GraphBase::AddUnary(const op::Unary* unary) {
        dawn::ErrorLog() << "Unimplemented: GraphBase::AddUnary";
        return {};
    }
    
    MaybeError GraphBase::Finish() {
        dawn::ErrorLog() << "Unimplemented: GraphBase::Finish";
        return {};
    }

    MaybeError GraphBase::Compile() {
        return CompileImpl();
    }

    MaybeError GraphBase::CompileImpl() {
        dawn::ErrorLog() << "Unimplemented: GraphBase::CompileImpl";
        return {};
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
