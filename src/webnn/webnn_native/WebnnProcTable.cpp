
#include "webnn/webnn_native/webnn_platform.h"
#include "webnn/webnn_native/WebnnNative.h"

#include <algorithm>
#include <vector>

#include "webnn/webnn_native/Context.h"
#include "webnn/webnn_native/Graph.h"
#include "webnn/webnn_native/GraphBuilder.h"
#include "webnn/webnn_native/NamedInputs.h"
#include "webnn/webnn_native/NamedOperands.h"
#include "webnn/webnn_native/NamedOutputs.h"
#include "webnn/webnn_native/Operand.h"
#include "webnn/webnn_native/OperandArray.h"
#include "webnn/webnn_native/Operator.h"
#include "webnn/webnn_native/OperatorArray.h"

namespace webnn_native {

    namespace {


        bool NativeContextPopErrorScope(MLContext cSelf, MLErrorCallback callback, void * userdata) {
            auto self = reinterpret_cast<ContextBase*>(cSelf);

            auto callback_ = callback;
            auto userdata_ = reinterpret_cast<void * >(userdata);
            auto result =            self->PopErrorScope(callback_, userdata_);
            return result;
        }

        void NativeContextPushErrorScope(MLContext cSelf, MLErrorFilter filter) {
            auto self = reinterpret_cast<ContextBase*>(cSelf);

            auto filter_ = static_cast<ml::ErrorFilter>(filter);
            self->PushErrorScope(filter_);
        }

        void NativeContextSetUncapturedErrorCallback(MLContext cSelf, MLErrorCallback callback, void * userdata) {
            auto self = reinterpret_cast<ContextBase*>(cSelf);

            auto callback_ = callback;
            auto userdata_ = reinterpret_cast<void * >(userdata);
            self->SetUncapturedErrorCallback(callback_, userdata_);
        }

        void NativeContextReference(MLContext cSelf) {
            auto self = reinterpret_cast<ContextBase*>(cSelf);

            self->Reference();
        }

        void NativeContextRelease(MLContext cSelf) {
            auto self = reinterpret_cast<ContextBase*>(cSelf);

            self->Release();
        }

        MLComputeGraphStatus NativeGraphCompute(MLGraph cSelf, MLNamedInputs inputs, MLNamedOutputs outputs) {
            auto self = reinterpret_cast<GraphBase*>(cSelf);

            auto inputs_ = reinterpret_cast<NamedInputsBase* >(inputs);
            auto outputs_ = reinterpret_cast<NamedOutputsBase* >(outputs);
            auto result =            self->Compute(inputs_, outputs_);
            return result;
        }

        void NativeGraphReference(MLGraph cSelf) {
            auto self = reinterpret_cast<GraphBase*>(cSelf);

            self->Reference();
        }

        void NativeGraphRelease(MLGraph cSelf) {
            auto self = reinterpret_cast<GraphBase*>(cSelf);

            self->Release();
        }

        MLOperand NativeGraphBuilderAbs(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Abs(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderAdd(MLGraphBuilder cSelf, MLOperand a, MLOperand b) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Add(a_, b_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderAveragePool2d(MLGraphBuilder cSelf, MLOperand input, MLPool2dOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<Pool2dOptions const * >(options);
            auto result =            self->AveragePool2d(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderBatchNorm(MLGraphBuilder cSelf, MLOperand input, MLOperand mean, MLOperand variance, MLBatchNormOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto mean_ = reinterpret_cast<OperandBase* >(mean);
            auto variance_ = reinterpret_cast<OperandBase* >(variance);
            auto options_ = reinterpret_cast<BatchNormOptions const * >(options);
            auto result =            self->BatchNorm(input_, mean_, variance_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLGraph NativeGraphBuilderBuild(MLGraphBuilder cSelf, MLNamedOperands namedOperands) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto namedOperands_ = reinterpret_cast<NamedOperandsBase* >(namedOperands);
            auto result =            self->Build(namedOperands_);
            return reinterpret_cast<MLGraph>(result);
        }

        MLOperand NativeGraphBuilderCeil(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Ceil(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderClamp(MLGraphBuilder cSelf, MLOperand input, MLClampOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ClampOptions const * >(options);
            auto result =            self->Clamp(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperator NativeGraphBuilderClampOperator(MLGraphBuilder cSelf, MLClampOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto options_ = reinterpret_cast<ClampOptions const * >(options);
            auto result =            self->ClampOperator(options_);
            return reinterpret_cast<MLOperator>(result);
        }

        MLOperand NativeGraphBuilderConcat(MLGraphBuilder cSelf, uint32_t inputsCount, MLOperand const * inputs, uint32_t axis) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto inputsCount_ = inputsCount;
            auto inputs_ = reinterpret_cast<OperandBase* const * >(inputs);
            auto axis_ = axis;
            auto result =            self->Concat(inputsCount_, inputs_, axis_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderConstant(MLGraphBuilder cSelf, MLOperandDescriptor const * desc, MLArrayBufferView const * value) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto desc_ = reinterpret_cast<OperandDescriptor const * >(desc);
            auto value_ = reinterpret_cast<ArrayBufferView const * >(value);
            auto result =            self->Constant(desc_, value_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderConv2d(MLGraphBuilder cSelf, MLOperand input, MLOperand filter, MLConv2dOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto filter_ = reinterpret_cast<OperandBase* >(filter);
            auto options_ = reinterpret_cast<Conv2dOptions const * >(options);
            auto result =            self->Conv2d(input_, filter_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderCos(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Cos(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderDiv(MLGraphBuilder cSelf, MLOperand a, MLOperand b) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Div(a_, b_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderExp(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Exp(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderFloor(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Floor(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderGemm(MLGraphBuilder cSelf, MLOperand a, MLOperand b, MLGemmOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto options_ = reinterpret_cast<GemmOptions const * >(options);
            auto result =            self->Gemm(a_, b_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperandArray NativeGraphBuilderGru(MLGraphBuilder cSelf, MLOperand input, MLOperand weight, MLOperand recurrentWeight, int32_t steps, int32_t hiddenSize, MLGruOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto weight_ = reinterpret_cast<OperandBase* >(weight);
            auto recurrentWeight_ = reinterpret_cast<OperandBase* >(recurrentWeight);
            auto steps_ = steps;
            auto hiddenSize_ = hiddenSize;
            auto options_ = reinterpret_cast<GruOptions const * >(options);
            auto result =            self->Gru(input_, weight_, recurrentWeight_, steps_, hiddenSize_, options_);
            return reinterpret_cast<MLOperandArray>(result);
        }

        MLOperand NativeGraphBuilderHardSwish(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->HardSwish(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperator NativeGraphBuilderHardSwishOperator(MLGraphBuilder cSelf) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto result =            self->HardSwishOperator();
            return reinterpret_cast<MLOperator>(result);
        }

        MLOperand NativeGraphBuilderInput(MLGraphBuilder cSelf, char const * name, MLOperandDescriptor const * desc) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto name_ = reinterpret_cast<char const * >(name);
            auto desc_ = reinterpret_cast<OperandDescriptor const * >(desc);
            auto result =            self->Input(name_, desc_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderInstanceNorm(MLGraphBuilder cSelf, MLOperand input, MLInstanceNormOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<InstanceNormOptions const * >(options);
            auto result =            self->InstanceNorm(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderLeakyRelu(MLGraphBuilder cSelf, MLOperand input, MLLeakyReluOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<LeakyReluOptions const * >(options);
            auto result =            self->LeakyRelu(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperator NativeGraphBuilderLeakyReluOperator(MLGraphBuilder cSelf, MLLeakyReluOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto options_ = reinterpret_cast<LeakyReluOptions const * >(options);
            auto result =            self->LeakyReluOperator(options_);
            return reinterpret_cast<MLOperator>(result);
        }

        MLOperand NativeGraphBuilderLog(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Log(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderMatmul(MLGraphBuilder cSelf, MLOperand a, MLOperand b) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Matmul(a_, b_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderMax(MLGraphBuilder cSelf, MLOperand a, MLOperand b) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Max(a_, b_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderMaxPool2d(MLGraphBuilder cSelf, MLOperand input, MLPool2dOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<Pool2dOptions const * >(options);
            auto result =            self->MaxPool2d(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderMin(MLGraphBuilder cSelf, MLOperand a, MLOperand b) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Min(a_, b_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderMul(MLGraphBuilder cSelf, MLOperand a, MLOperand b) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Mul(a_, b_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderNeg(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Neg(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderPad(MLGraphBuilder cSelf, MLOperand input, MLOperand padding, MLPadOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto padding_ = reinterpret_cast<OperandBase* >(padding);
            auto options_ = reinterpret_cast<PadOptions const * >(options);
            auto result =            self->Pad(input_, padding_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderPow(MLGraphBuilder cSelf, MLOperand a, MLOperand b) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Pow(a_, b_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderReduceL1(MLGraphBuilder cSelf, MLOperand input, MLReduceOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ReduceOptions const * >(options);
            auto result =            self->ReduceL1(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderReduceL2(MLGraphBuilder cSelf, MLOperand input, MLReduceOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ReduceOptions const * >(options);
            auto result =            self->ReduceL2(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderReduceMax(MLGraphBuilder cSelf, MLOperand input, MLReduceOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ReduceOptions const * >(options);
            auto result =            self->ReduceMax(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderReduceMean(MLGraphBuilder cSelf, MLOperand input, MLReduceOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ReduceOptions const * >(options);
            auto result =            self->ReduceMean(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderReduceMin(MLGraphBuilder cSelf, MLOperand input, MLReduceOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ReduceOptions const * >(options);
            auto result =            self->ReduceMin(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderReduceProduct(MLGraphBuilder cSelf, MLOperand input, MLReduceOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ReduceOptions const * >(options);
            auto result =            self->ReduceProduct(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderReduceSum(MLGraphBuilder cSelf, MLOperand input, MLReduceOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ReduceOptions const * >(options);
            auto result =            self->ReduceSum(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderRelu(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Relu(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperator NativeGraphBuilderReluOperator(MLGraphBuilder cSelf) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto result =            self->ReluOperator();
            return reinterpret_cast<MLOperator>(result);
        }

        MLOperand NativeGraphBuilderResample(MLGraphBuilder cSelf, MLOperand input, MLResampleOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ResampleOptions const * >(options);
            auto result =            self->Resample(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderReshape(MLGraphBuilder cSelf, MLOperand input, int32_t const * newShape, uint32_t newShapeCount) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto newShape_ = reinterpret_cast<int32_t const * >(newShape);
            auto newShapeCount_ = newShapeCount;
            auto result =            self->Reshape(input_, newShape_, newShapeCount_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderSigmoid(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Sigmoid(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperator NativeGraphBuilderSigmoidOperator(MLGraphBuilder cSelf) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto result =            self->SigmoidOperator();
            return reinterpret_cast<MLOperator>(result);
        }

        MLOperand NativeGraphBuilderSin(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Sin(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderSlice(MLGraphBuilder cSelf, MLOperand input, int32_t const * starts, uint32_t startsCount, int32_t const * sizes, uint32_t sizesCount, MLSliceOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto starts_ = reinterpret_cast<int32_t const * >(starts);
            auto startsCount_ = startsCount;
            auto sizes_ = reinterpret_cast<int32_t const * >(sizes);
            auto sizesCount_ = sizesCount;
            auto options_ = reinterpret_cast<SliceOptions const * >(options);
            auto result =            self->Slice(input_, starts_, startsCount_, sizes_, sizesCount_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderSoftmax(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Softmax(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperandArray NativeGraphBuilderSplit(MLGraphBuilder cSelf, MLOperand input, uint32_t const * splits, uint32_t splitsCount, MLSplitOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto splits_ = reinterpret_cast<uint32_t const * >(splits);
            auto splitsCount_ = splitsCount;
            auto options_ = reinterpret_cast<SplitOptions const * >(options);
            auto result =            self->Split(input_, splits_, splitsCount_, options_);
            return reinterpret_cast<MLOperandArray>(result);
        }

        MLOperand NativeGraphBuilderSqueeze(MLGraphBuilder cSelf, MLOperand input, MLSqueezeOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<SqueezeOptions const * >(options);
            auto result =            self->Squeeze(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderSub(MLGraphBuilder cSelf, MLOperand a, MLOperand b) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Sub(a_, b_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderTan(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Tan(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperand NativeGraphBuilderTanh(MLGraphBuilder cSelf, MLOperand input) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Tanh(input_);
            return reinterpret_cast<MLOperand>(result);
        }

        MLOperator NativeGraphBuilderTanhOperator(MLGraphBuilder cSelf) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto result =            self->TanhOperator();
            return reinterpret_cast<MLOperator>(result);
        }

        MLOperand NativeGraphBuilderTranspose(MLGraphBuilder cSelf, MLOperand input, MLTransposeOptions const * options) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<TransposeOptions const * >(options);
            auto result =            self->Transpose(input_, options_);
            return reinterpret_cast<MLOperand>(result);
        }

        void NativeGraphBuilderReference(MLGraphBuilder cSelf) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            self->Reference();
        }

        void NativeGraphBuilderRelease(MLGraphBuilder cSelf) {
            auto self = reinterpret_cast<GraphBuilderBase*>(cSelf);

            self->Release();
        }

        void NativeNamedInputsSet(MLNamedInputs cSelf, char const * name, MLInput const * input) {
            auto self = reinterpret_cast<NamedInputsBase*>(cSelf);

            auto name_ = reinterpret_cast<char const * >(name);
            auto input_ = reinterpret_cast<Input const * >(input);
            self->Set(name_, input_);
        }

        void NativeNamedInputsReference(MLNamedInputs cSelf) {
            auto self = reinterpret_cast<NamedInputsBase*>(cSelf);

            self->Reference();
        }

        void NativeNamedInputsRelease(MLNamedInputs cSelf) {
            auto self = reinterpret_cast<NamedInputsBase*>(cSelf);

            self->Release();
        }

        void NativeNamedOperandsSet(MLNamedOperands cSelf, char const * name, MLOperand operand) {
            auto self = reinterpret_cast<NamedOperandsBase*>(cSelf);

            auto name_ = reinterpret_cast<char const * >(name);
            auto operand_ = reinterpret_cast<OperandBase* >(operand);
            self->Set(name_, operand_);
        }

        void NativeNamedOperandsReference(MLNamedOperands cSelf) {
            auto self = reinterpret_cast<NamedOperandsBase*>(cSelf);

            self->Reference();
        }

        void NativeNamedOperandsRelease(MLNamedOperands cSelf) {
            auto self = reinterpret_cast<NamedOperandsBase*>(cSelf);

            self->Release();
        }

        void NativeNamedOutputsSet(MLNamedOutputs cSelf, char const * name, MLArrayBufferView const * resource) {
            auto self = reinterpret_cast<NamedOutputsBase*>(cSelf);

            auto name_ = reinterpret_cast<char const * >(name);
            auto resource_ = reinterpret_cast<ArrayBufferView const * >(resource);
            self->Set(name_, resource_);
        }

        void NativeNamedOutputsReference(MLNamedOutputs cSelf) {
            auto self = reinterpret_cast<NamedOutputsBase*>(cSelf);

            self->Reference();
        }

        void NativeNamedOutputsRelease(MLNamedOutputs cSelf) {
            auto self = reinterpret_cast<NamedOutputsBase*>(cSelf);

            self->Release();
        }

        void NativeOperandReference(MLOperand cSelf) {
            auto self = reinterpret_cast<OperandBase*>(cSelf);

            self->Reference();
        }

        void NativeOperandRelease(MLOperand cSelf) {
            auto self = reinterpret_cast<OperandBase*>(cSelf);

            self->Release();
        }

        MLOperand NativeOperandArrayGetOperand(MLOperandArray cSelf, size_t index) {
            auto self = reinterpret_cast<OperandArrayBase*>(cSelf);

            auto index_ = index;
            auto result =            self->GetOperand(index_);
            return reinterpret_cast<MLOperand>(result);
        }

        size_t NativeOperandArraySize(MLOperandArray cSelf) {
            auto self = reinterpret_cast<OperandArrayBase*>(cSelf);

            auto result =            self->Size();
            return result;
        }

        void NativeOperandArrayReference(MLOperandArray cSelf) {
            auto self = reinterpret_cast<OperandArrayBase*>(cSelf);

            self->Reference();
        }

        void NativeOperandArrayRelease(MLOperandArray cSelf) {
            auto self = reinterpret_cast<OperandArrayBase*>(cSelf);

            self->Release();
        }

        void NativeOperatorReference(MLOperator cSelf) {
            auto self = reinterpret_cast<OperatorBase*>(cSelf);

            self->Reference();
        }

        void NativeOperatorRelease(MLOperator cSelf) {
            auto self = reinterpret_cast<OperatorBase*>(cSelf);

            self->Release();
        }

        MLOperator NativeOperatorArrayGetOperator(MLOperatorArray cSelf, size_t index) {
            auto self = reinterpret_cast<OperatorArrayBase*>(cSelf);

            auto index_ = index;
            auto result =            self->GetOperator(index_);
            return reinterpret_cast<MLOperator>(result);
        }

        void NativeOperatorArraySet(MLOperatorArray cSelf, MLOperator op) {
            auto self = reinterpret_cast<OperatorArrayBase*>(cSelf);

            auto op_ = reinterpret_cast<OperatorBase* >(op);
            self->Set(op_);
        }

        size_t NativeOperatorArraySize(MLOperatorArray cSelf) {
            auto self = reinterpret_cast<OperatorArrayBase*>(cSelf);

            auto result =            self->Size();
            return result;
        }

        void NativeOperatorArrayReference(MLOperatorArray cSelf) {
            auto self = reinterpret_cast<OperatorArrayBase*>(cSelf);

            self->Reference();
        }

        void NativeOperatorArrayRelease(MLOperatorArray cSelf) {
            auto self = reinterpret_cast<OperatorArrayBase*>(cSelf);

            self->Release();
        }

        struct ProcEntry {
            MLProc proc;
            const char* name;
        };
        static const ProcEntry sProcMap[] = {
            { reinterpret_cast<MLProc>(NativeContextPopErrorScope), "mlContextPopErrorScope" },
            { reinterpret_cast<MLProc>(NativeContextPushErrorScope), "mlContextPushErrorScope" },
            { reinterpret_cast<MLProc>(NativeContextReference), "mlContextReference" },
            { reinterpret_cast<MLProc>(NativeContextRelease), "mlContextRelease" },
            { reinterpret_cast<MLProc>(NativeContextSetUncapturedErrorCallback), "mlContextSetUncapturedErrorCallback" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderAbs), "mlGraphBuilderAbs" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderAdd), "mlGraphBuilderAdd" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderAveragePool2d), "mlGraphBuilderAveragePool2d" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderBatchNorm), "mlGraphBuilderBatchNorm" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderBuild), "mlGraphBuilderBuild" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderCeil), "mlGraphBuilderCeil" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderClamp), "mlGraphBuilderClamp" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderClampOperator), "mlGraphBuilderClampOperator" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderConcat), "mlGraphBuilderConcat" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderConstant), "mlGraphBuilderConstant" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderConv2d), "mlGraphBuilderConv2d" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderCos), "mlGraphBuilderCos" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderDiv), "mlGraphBuilderDiv" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderExp), "mlGraphBuilderExp" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderFloor), "mlGraphBuilderFloor" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderGemm), "mlGraphBuilderGemm" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderGru), "mlGraphBuilderGru" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderHardSwish), "mlGraphBuilderHardSwish" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderHardSwishOperator), "mlGraphBuilderHardSwishOperator" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderInput), "mlGraphBuilderInput" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderInstanceNorm), "mlGraphBuilderInstanceNorm" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderLeakyRelu), "mlGraphBuilderLeakyRelu" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderLeakyReluOperator), "mlGraphBuilderLeakyReluOperator" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderLog), "mlGraphBuilderLog" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderMatmul), "mlGraphBuilderMatmul" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderMax), "mlGraphBuilderMax" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderMaxPool2d), "mlGraphBuilderMaxPool2d" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderMin), "mlGraphBuilderMin" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderMul), "mlGraphBuilderMul" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderNeg), "mlGraphBuilderNeg" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderPad), "mlGraphBuilderPad" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderPow), "mlGraphBuilderPow" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReduceL1), "mlGraphBuilderReduceL1" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReduceL2), "mlGraphBuilderReduceL2" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReduceMax), "mlGraphBuilderReduceMax" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReduceMean), "mlGraphBuilderReduceMean" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReduceMin), "mlGraphBuilderReduceMin" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReduceProduct), "mlGraphBuilderReduceProduct" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReduceSum), "mlGraphBuilderReduceSum" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReference), "mlGraphBuilderReference" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderRelease), "mlGraphBuilderRelease" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderRelu), "mlGraphBuilderRelu" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReluOperator), "mlGraphBuilderReluOperator" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderResample), "mlGraphBuilderResample" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderReshape), "mlGraphBuilderReshape" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderSigmoid), "mlGraphBuilderSigmoid" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderSigmoidOperator), "mlGraphBuilderSigmoidOperator" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderSin), "mlGraphBuilderSin" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderSlice), "mlGraphBuilderSlice" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderSoftmax), "mlGraphBuilderSoftmax" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderSplit), "mlGraphBuilderSplit" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderSqueeze), "mlGraphBuilderSqueeze" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderSub), "mlGraphBuilderSub" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderTan), "mlGraphBuilderTan" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderTanh), "mlGraphBuilderTanh" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderTanhOperator), "mlGraphBuilderTanhOperator" },
            { reinterpret_cast<MLProc>(NativeGraphBuilderTranspose), "mlGraphBuilderTranspose" },
            { reinterpret_cast<MLProc>(NativeGraphCompute), "mlGraphCompute" },
            { reinterpret_cast<MLProc>(NativeGraphReference), "mlGraphReference" },
            { reinterpret_cast<MLProc>(NativeGraphRelease), "mlGraphRelease" },
            { reinterpret_cast<MLProc>(NativeNamedInputsReference), "mlNamedInputsReference" },
            { reinterpret_cast<MLProc>(NativeNamedInputsRelease), "mlNamedInputsRelease" },
            { reinterpret_cast<MLProc>(NativeNamedInputsSet), "mlNamedInputsSet" },
            { reinterpret_cast<MLProc>(NativeNamedOperandsReference), "mlNamedOperandsReference" },
            { reinterpret_cast<MLProc>(NativeNamedOperandsRelease), "mlNamedOperandsRelease" },
            { reinterpret_cast<MLProc>(NativeNamedOperandsSet), "mlNamedOperandsSet" },
            { reinterpret_cast<MLProc>(NativeNamedOutputsReference), "mlNamedOutputsReference" },
            { reinterpret_cast<MLProc>(NativeNamedOutputsRelease), "mlNamedOutputsRelease" },
            { reinterpret_cast<MLProc>(NativeNamedOutputsSet), "mlNamedOutputsSet" },
            { reinterpret_cast<MLProc>(NativeOperandArrayGetOperand), "mlOperandArrayGetOperand" },
            { reinterpret_cast<MLProc>(NativeOperandArrayReference), "mlOperandArrayReference" },
            { reinterpret_cast<MLProc>(NativeOperandArrayRelease), "mlOperandArrayRelease" },
            { reinterpret_cast<MLProc>(NativeOperandArraySize), "mlOperandArraySize" },
            { reinterpret_cast<MLProc>(NativeOperandReference), "mlOperandReference" },
            { reinterpret_cast<MLProc>(NativeOperandRelease), "mlOperandRelease" },
            { reinterpret_cast<MLProc>(NativeOperatorArrayGetOperator), "mlOperatorArrayGetOperator" },
            { reinterpret_cast<MLProc>(NativeOperatorArrayReference), "mlOperatorArrayReference" },
            { reinterpret_cast<MLProc>(NativeOperatorArrayRelease), "mlOperatorArrayRelease" },
            { reinterpret_cast<MLProc>(NativeOperatorArraySet), "mlOperatorArraySet" },
            { reinterpret_cast<MLProc>(NativeOperatorArraySize), "mlOperatorArraySize" },
            { reinterpret_cast<MLProc>(NativeOperatorReference), "mlOperatorReference" },
            { reinterpret_cast<MLProc>(NativeOperatorRelease), "mlOperatorRelease" },
        };
        static constexpr size_t sProcMapSize = sizeof(sProcMap) / sizeof(sProcMap[0]);
    }

    std::vector<const char*> GetProcMapNamesForTestingInternal() {
        std::vector<const char*> result;
        result.reserve(sProcMapSize);
        for (const ProcEntry& entry : sProcMap) {
            result.push_back(entry.name);
        }
        return result;
    }

    MLGraphBuilder NativeCreateGraphBuilder(MLContext context) {
        return reinterpret_cast<MLGraphBuilder>(new GraphBuilderBase(reinterpret_cast<ContextBase *>(context)));
    }

    MLNamedInputs NativeCreateNamedInputs() {
        return reinterpret_cast<MLNamedInputs>(new NamedInputsBase());
    }

    MLNamedOperands NativeCreateNamedOperands() {
         return reinterpret_cast<MLNamedOperands>(new NamedOperandsBase());
    }

    MLNamedOutputs NativeCreateNamedOutputs() {
         return reinterpret_cast<MLNamedOutputs>(new NamedOutputsBase());
    }

    MLOperatorArray NativeCreateOperatorArray() {
         return reinterpret_cast<MLOperatorArray>(new OperatorArrayBase());
    }

    static WebnnProcTable gProcTable = {
        NativeCreateGraphBuilder,
        NativeCreateNamedInputs,
        NativeCreateNamedOperands,
        NativeCreateNamedOutputs,
        NativeCreateOperatorArray,
        NativeContextPopErrorScope,
        NativeContextPushErrorScope,
        NativeContextSetUncapturedErrorCallback,
        NativeContextReference,
        NativeContextRelease,
        NativeGraphCompute,
        NativeGraphReference,
        NativeGraphRelease,
        NativeGraphBuilderAbs,
        NativeGraphBuilderAdd,
        NativeGraphBuilderAveragePool2d,
        NativeGraphBuilderBatchNorm,
        NativeGraphBuilderBuild,
        NativeGraphBuilderCeil,
        NativeGraphBuilderClamp,
        NativeGraphBuilderClampOperator,
        NativeGraphBuilderConcat,
        NativeGraphBuilderConstant,
        NativeGraphBuilderConv2d,
        NativeGraphBuilderCos,
        NativeGraphBuilderDiv,
        NativeGraphBuilderExp,
        NativeGraphBuilderFloor,
        NativeGraphBuilderGemm,
        NativeGraphBuilderGru,
        NativeGraphBuilderHardSwish,
        NativeGraphBuilderHardSwishOperator,
        NativeGraphBuilderInput,
        NativeGraphBuilderInstanceNorm,
        NativeGraphBuilderLeakyRelu,
        NativeGraphBuilderLeakyReluOperator,
        NativeGraphBuilderLog,
        NativeGraphBuilderMatmul,
        NativeGraphBuilderMax,
        NativeGraphBuilderMaxPool2d,
        NativeGraphBuilderMin,
        NativeGraphBuilderMul,
        NativeGraphBuilderNeg,
        NativeGraphBuilderPad,
        NativeGraphBuilderPow,
        NativeGraphBuilderReduceL1,
        NativeGraphBuilderReduceL2,
        NativeGraphBuilderReduceMax,
        NativeGraphBuilderReduceMean,
        NativeGraphBuilderReduceMin,
        NativeGraphBuilderReduceProduct,
        NativeGraphBuilderReduceSum,
        NativeGraphBuilderRelu,
        NativeGraphBuilderReluOperator,
        NativeGraphBuilderResample,
        NativeGraphBuilderReshape,
        NativeGraphBuilderSigmoid,
        NativeGraphBuilderSigmoidOperator,
        NativeGraphBuilderSin,
        NativeGraphBuilderSlice,
        NativeGraphBuilderSoftmax,
        NativeGraphBuilderSplit,
        NativeGraphBuilderSqueeze,
        NativeGraphBuilderSub,
        NativeGraphBuilderTan,
        NativeGraphBuilderTanh,
        NativeGraphBuilderTanhOperator,
        NativeGraphBuilderTranspose,
        NativeGraphBuilderReference,
        NativeGraphBuilderRelease,
        NativeNamedInputsSet,
        NativeNamedInputsReference,
        NativeNamedInputsRelease,
        NativeNamedOperandsSet,
        NativeNamedOperandsReference,
        NativeNamedOperandsRelease,
        NativeNamedOutputsSet,
        NativeNamedOutputsReference,
        NativeNamedOutputsRelease,
        NativeOperandReference,
        NativeOperandRelease,
        NativeOperandArrayGetOperand,
        NativeOperandArraySize,
        NativeOperandArrayReference,
        NativeOperandArrayRelease,
        NativeOperatorReference,
        NativeOperatorRelease,
        NativeOperatorArrayGetOperator,
        NativeOperatorArraySet,
        NativeOperatorArraySize,
        NativeOperatorArrayReference,
        NativeOperatorArrayRelease,
    };

    const WebnnProcTable& GetProcsAutogen() {
        return gProcTable;
    }

}
