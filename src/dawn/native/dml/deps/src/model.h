//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include "dawn/common/Assert.h"

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
        TensorData(dawn::native::d3d12::Buffer* buffer,
                   size_t size,
                   size_t offset = 0) :
            buffer(buffer),
            size(size),
            offset(offset) {}

        dawn::native::d3d12::Buffer* Get() const { return buffer; }

        size_t Size() const { return size; }

        size_t Offset() const { return offset; }

        dawn::native::d3d12::Buffer* buffer;
        size_t size;
        size_t offset;
    };

    struct Binding
    {
        explicit Binding(dml::Expression& expression, 
                         dawn::native::d3d12::Buffer * buffer,
                         size_t size,
                         size_t offset = 0)
            :   exp(expression),
                desc(expression.GetOutputDesc()),
                data(buffer, size, offset)
        {}

        dml::Expression exp;
        dml::TensorDesc desc;
        TensorData data;
    };
}
