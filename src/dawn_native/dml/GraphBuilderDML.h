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

#ifndef WEBNN_NATIVE_DML_GRAPH_BUILDER_DML_H_
#define WEBNN_NATIVE_DML_GRAPH_BUILDER_DML_H_

#include "dawn_native/GraphBuilder.h"

namespace dawn::native { namespace dml {

    class GraphBuilder : public GraphBuilderBase {
      public:
        static GraphBuilder* Create(DeviceBase* device);

      private:
        GraphBuilder(DeviceBase* context);
        GraphBuilder(DeviceBase* device, ObjectBase::ErrorTag tag);
        virtual ~GraphBuilder() = default;

        virtual bool InitializeImpl() override;
        virtual GraphBase* CreateGraphImpl() override;
    };

}}  // namespace dawn::native::dml

#endif  // WEBNN_NATIVE_DML_GRAPH_BUILDER_DML_H_
