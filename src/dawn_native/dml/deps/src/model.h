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
        TensorData(dawn_native::d3d12::Buffer* buffer,
                   size_t size,
                   size_t offset = 0) :
            buffer(AcquireRef(buffer)),
            size(size),
            offset(offset) {}

        dawn_native::d3d12::Buffer* Get() const { return buffer.Get(); }

        size_t Size() const { return size; }

        size_t Offset() const { return offset; }

        Ref<dawn_native::d3d12::Buffer> buffer;
        size_t size;
        size_t offset;
    };

    struct Binding
    {
        explicit Binding(dml::Expression& expression, 
                         dawn_native::d3d12::Buffer * buffer,
                         size_t size,
                         size_t offset = 0)
            :   desc(expression.GetOutputDesc()),
                data(buffer, size, offset)
        {}

        dml::TensorDesc desc;
        TensorData data;
    };
}
