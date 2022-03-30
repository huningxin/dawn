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

#ifndef WEBNN_NATIVE_OPS_BATCHNORM_H_
#define WEBNN_NATIVE_OPS_BATCHNORM_H_

#include "dawn/native/FusionOperator.h"
#include "dawn/native/Graph.h"
#include "dawn/native/Operand.h"
#include "dawn/native/Operator.h"

namespace dawn::native { namespace op {

    class BatchNorm final : public OperatorBase {
      public:
        BatchNorm(GraphBuilderBase* builder,
                  OperandBase* input,
                  OperandBase* mean,
                  OperandBase* variance,
                  BatchNormOptions const* options);
        ~BatchNorm() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddBatchNorm(this);
        }
        MaybeError ValidateAndInferOutputInfo() override;

        BatchNormOptions const* GetOptions() const {
            return &mOptions;
        }

      private:
        BatchNormOptions mOptions;
        Ref<FusionOperatorBase> mActivation;
    };

}}  // namespace dawn::native::op

#endif  // WEBNN_NATIVE_OPS_BATCHNORM_H_
