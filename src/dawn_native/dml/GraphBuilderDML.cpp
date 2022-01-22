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

#include "dawn_native/dml/GraphBuilderDML.h"

#include "common/RefCounted.h"
#include "dawn_native/dml/GraphDML.h"

namespace dawn_native { namespace dml {

    // static
    GraphBuilder* GraphBuilder::Create(DeviceBase* device) {
        return new GraphBuilder(device);
    }

    GraphBuilder::GraphBuilder(DeviceBase* device) : GraphBuilderBase(device) {
    }

    bool GraphBuilder::InitializeImpl() {
        return true;
    }

    GraphBase* GraphBuilder::CreateGraphImpl() {
        return new Graph(GetDevice());
    }

}}  // namespace dawn_native::dml
