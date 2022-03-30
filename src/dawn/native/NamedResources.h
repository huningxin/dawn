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

#ifndef WEBNN_NATIVE_NAMED_RESOURCES_H_
#define WEBNN_NATIVE_NAMED_RESOURCES_H_

#include <map>
#include <string>

#include "dawn/native/NamedRecords.h"
#include "dawn/native/dawn_platform.h"

namespace dawn::native {

    class NamedResourcesBase : public RefCounted {
      public:
        NamedResourcesBase() = default;
        virtual ~NamedResourcesBase() = default;

        // WebNN API
        void APISet(char const* name, const BufferResourceView* record) {
            mResources[std::string(name)] = *record;
        }

        // Other methods
        const std::map<std::string, BufferResourceView>& GetResources() const {
            return mResources;
        }

      private:
        std::map<std::string, BufferResourceView> mResources;
    };

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_NAMED_RESOURCES_H_
