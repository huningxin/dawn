//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#define NOMINMAX
#include <cassert>
#include <optional>
#include <string>
#include <functional>
#include <numeric>

#ifdef __cpp_lib_span
#include <span>
#endif

// #include <Windows.h>
// #include <d3d12.h>

// // ToDo: dxgi isn't available in WSL.
// #include <dxgi1_5.h>
// #include <dxgidebug.h>

// #include <initguid.h>
// #include <wrl/client.h>
// #include <wrl/implements.h>

#include "dawn_native/d3d12/d3d12_platform.h"
#include "dawn_native/d3d12/DeviceD3D12.h"
#include "dawn_native/d3d12/BufferD3D12.h"
#include "dawn_native/d3d12/CommandRecordingContext.h"
#include "dawn_native/Error.h"
#include "dawn_native/ErrorData.h"

#define DML_TARGET_VERSION_USE_LATEST 1
#include <DirectML.h>
#include <DirectMLX.h>

#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS
#include "third_party/DirectML/Python/src/d3dx12.h"
#include "util.h"
#include "model.h"
#include "typeconvert.h"
#include "dmldevice.h"
