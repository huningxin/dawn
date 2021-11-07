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

#ifndef WEBNN_NATIVE_OPERAND_H_
#define WEBNN_NATIVE_OPERAND_H_

#include <string>
#include <vector>

#include "dawn_native/webnn/Forward.h"
#include "dawn_native/webnn/Graph.h"
#include "dawn_native/webnn/ObjectBase.h"
#include "dawn_native/webnn/Operator.h"
#include "dawn_native/webnn/webnn_platform.h"

namespace webnn_native {

    class OperandBase : public ObjectBase {
      public:
        OperandBase(GraphBuilderBase*, OperatorBase*);
        virtual ~OperandBase() = default;

        const OperatorBase* Operator() {
            return mOperator.Get();
        }

        ml::OperandType Type() const {
            return mType;
        }
        void SetType(ml::OperandType type) {
            mType = type;
        }
        uint32_t Rank() const {
            return mRank;
        }
        void SetRank(uint32_t rank) {
            mRank = rank;
        }

        static OperandBase* MakeError(GraphBuilderBase* modelBuilder);

      private:
        OperandBase(GraphBuilderBase* GraphBuilder, ObjectBase::ErrorTag tag);

      protected:
        // The operator of generating the operand.
        Ref<OperatorBase> mOperator;
        // The operand type.
        ml::OperandType mType;
        // only set rank for dimensions
        uint32_t mRank;
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_OPERAND_H_
