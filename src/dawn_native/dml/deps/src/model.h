//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include "common/Assert.h"

namespace pydml
{
    struct CompiledModel
    {
        CompiledModel(
            dml::Graph& graph, 
            DML_EXECUTION_FLAGS flags,
            std::vector<dml::Expression>& outputs
            ) : 
            op(graph.Compile(flags, outputs))
        {}

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> op;
    };

    struct TensorData
    {
        TensorData(ID3D12Resource* buffer,
                   size_t size,
                   size_t offset = 0) :
            buffer(buffer),
            size(size),
            offset(offset) {}

        TensorData(dml::TensorDesc* desc) :
            size(desc->totalTensorSizeInBytes),
            desc(*desc->AsPtr<DML_BUFFER_TENSOR_DESC>())
        {
            DAWN_UNREACHABLE();
        }

        TensorData() {}

        ID3D12Resource* Get() const { return buffer; }

        size_t Size() const { return size; }

        size_t Offset() const { return offset; }

        const dml::TensorDesc* Desc() const { return &desc; }

        ID3D12Resource* buffer;
        size_t size;
        size_t offset;
        dml::TensorDesc desc;
    };

    struct Binding
    {
        explicit Binding(dml::Expression& expression, 
                         ID3D12Resource * buffer,
                         size_t size,
                         size_t offset = 0)
            :   desc(expression.GetOutputDesc()),
                data(buffer, size, offset)
        {}

        Binding() = default;

        dml::TensorDesc desc;
        TensorData data;
    };
}
