//
// DeviceResources.cpp - A wrapper for the Direct3D 12 device
//

#include "pch.h"
#include "DeviceResources.h"

using namespace DirectX;
using namespace DX;

using Microsoft::WRL::ComPtr;

namespace
{
    inline DXGI_FORMAT NoSRGB(DXGI_FORMAT fmt)
    {
        switch (fmt)
        {
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:   return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:   return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:   return DXGI_FORMAT_B8G8R8X8_UNORM;
        default:                                return fmt;
        }
    }
};

// Constructor for DeviceResources.
DeviceResources::DeviceResources(
    DXGI_FORMAT backBufferFormat,
    DXGI_FORMAT depthBufferFormat,
    UINT backBufferCount,
    D3D_FEATURE_LEVEL minFeatureLevel,
    unsigned int flags) noexcept(false) :
        m_backBufferIndex(0),
        m_fenceValues{},
        m_rtvDescriptorSize(0),
        m_screenViewport{},
        m_scissorRect{},
        m_backBufferFormat(backBufferFormat),
        m_depthBufferFormat(depthBufferFormat),
        m_backBufferCount(backBufferCount),
        m_d3dMinFeatureLevel(minFeatureLevel),
        m_window(nullptr),
        m_d3dFeatureLevel(D3D_FEATURE_LEVEL_11_0),
        m_dxgiFactoryFlags(0),
        m_outputSize{0, 0, 1, 1},
        m_colorSpace(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709),
        m_options(flags),
        m_deviceNotify(nullptr)
{
    if (backBufferCount > MAX_BACK_BUFFER_COUNT)
    {
        throw std::out_of_range("backBufferCount too large");
    }

    if (minFeatureLevel < D3D_FEATURE_LEVEL_11_0)
    {
        throw std::out_of_range("minFeatureLevel too low");
    }
}

// Destructor for DeviceResources.
DeviceResources::~DeviceResources()
{
    // Ensure that the GPU is no longer referencing resources that are about to be destroyed.
    WaitForGpu();
}

// Configures the Direct3D device, and stores handles to it and the device context.
void DeviceResources::CreateDeviceResources()
{
#if defined(_DEBUG)
    // Enable the debug layer (requires the Graphics Tools "optional feature").
    //
    // NOTE: Enabling the debug layer after device creation will invalidate the active device.
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(debugController.GetAddressOf()))))
        {
            debugController->EnableDebugLayer();
        }
        else
        {
            OutputDebugStringA("WARNING: Direct3D Debug Device is not available\n");
        }

        ComPtr<IDXGIInfoQueue> dxgiInfoQueue;
        if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(dxgiInfoQueue.GetAddressOf()))))
        {
            m_dxgiFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;

            /*dxgiInfoQueue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, true);
            dxgiInfoQueue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, true);*/

            DXGI_INFO_QUEUE_MESSAGE_ID hide[] =
            {
                80 /* IDXGISwapChain::GetContainingOutput: The swapchain's adapter does not control the output on which the swapchain's window resides. */,
            };
            DXGI_INFO_QUEUE_FILTER filter = {};
            filter.DenyList.NumIDs = _countof(hide);
            filter.DenyList.pIDList = hide;
            dxgiInfoQueue->AddStorageFilterEntries(DXGI_DEBUG_DXGI, &filter);
        }
    }
#endif

    ThrowIfFailed(CreateDXGIFactory2(m_dxgiFactoryFlags, IID_PPV_ARGS(m_dxgiFactory.ReleaseAndGetAddressOf())));

    ComPtr<IDXGIAdapter1> adapter;
    GetAdapter(adapter.GetAddressOf());

    // Create the DX12 API device object.
    ThrowIfFailed(D3D12CreateDevice(
        adapter.Get(),
        m_d3dMinFeatureLevel,
        IID_PPV_ARGS(m_d3dDevice.ReleaseAndGetAddressOf())
        ));

    m_d3dDevice->SetName(L"DeviceResources");

#ifndef NDEBUG
    // Configure debug device (if active).
    ComPtr<ID3D12InfoQueue> d3dInfoQueue;
    if (SUCCEEDED(m_d3dDevice.As(&d3dInfoQueue)))
    {
#ifdef _DEBUG
        /*d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
        d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);*/
#endif
        D3D12_MESSAGE_ID hide[] =
        {
            D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
            D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,
            D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_WRONGSWAPCHAINBUFFERREFERENCE
        };
        D3D12_INFO_QUEUE_FILTER filter = {};
        filter.DenyList.NumIDs = _countof(hide);
        filter.DenyList.pIDList = hide;
        d3dInfoQueue->AddStorageFilterEntries(&filter);
    }
#endif

    // Determine maximum supported feature level for this device
    static const D3D_FEATURE_LEVEL s_featureLevels[] =
    {
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };

    D3D12_FEATURE_DATA_FEATURE_LEVELS featLevels =
    {
        _countof(s_featureLevels), s_featureLevels, D3D_FEATURE_LEVEL_11_0
    };

    HRESULT hr = m_d3dDevice->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &featLevels, sizeof(featLevels));
    if (SUCCEEDED(hr))
    {
        m_d3dFeatureLevel = featLevels.MaxSupportedFeatureLevel;
    }
    else
    {
        m_d3dFeatureLevel = m_d3dMinFeatureLevel;
    }

    // Create the command queue.
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    ThrowIfFailed(m_d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(m_commandQueue.ReleaseAndGetAddressOf())));

    m_commandQueue->SetName(L"DeviceResources");

    // Create descriptor heaps for render target views and depth stencil views.
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = {};
    rtvDescriptorHeapDesc.NumDescriptors = m_backBufferCount;
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

    ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(m_rtvDescriptorHeap.ReleaseAndGetAddressOf())));

    m_rtvDescriptorHeap->SetName(L"DeviceResources");

    m_rtvDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    if (m_depthBufferFormat != DXGI_FORMAT_UNKNOWN)
    {
        D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc = {};
        dsvDescriptorHeapDesc.NumDescriptors = 1;
        dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;

        ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(m_dsvDescriptorHeap.ReleaseAndGetAddressOf())));

        m_dsvDescriptorHeap->SetName(L"DeviceResources");
    }

    // Create a command allocator for each back buffer that will be rendered to.
    for (UINT n = 0; n < m_backBufferCount; n++)
    {
        ThrowIfFailed(m_d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(m_commandAllocators[n].ReleaseAndGetAddressOf())));

        wchar_t name[25] = {};
        swprintf_s(name, L"Render target %u", n);
        m_commandAllocators[n]->SetName(name);
    }

    // Create a command list for recording graphics commands.
    ThrowIfFailed(m_d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocators[0].Get(), nullptr, IID_PPV_ARGS(m_commandList.ReleaseAndGetAddressOf())));
    ThrowIfFailed(m_commandList->Close());

    m_commandList->SetName(L"DeviceResources");

    // Create a fence for tracking GPU execution progress.
    ThrowIfFailed(m_d3dDevice->CreateFence(m_fenceValues[m_backBufferIndex], D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(m_fence.ReleaseAndGetAddressOf())));
    m_fenceValues[m_backBufferIndex]++;

    m_d3dDevice->CreateSharedHandle(m_fence.Get(),
        NULL,
        GENERIC_ALL,
        NULL,
        &m_shared_fence_handle);

    m_fence->SetName(L"DeviceResources");

    m_fenceEvent.Attach(CreateEventEx(nullptr, nullptr, 0, EVENT_MODIFY_STATE | SYNCHRONIZE));
    if (!m_fenceEvent.IsValid())
    {
        throw std::exception("CreateEvent");
    }
}

// Recreate all device resources and set them back to the current state.
void DeviceResources::HandleDeviceLost()
{
    if (m_deviceNotify)
    {
        m_deviceNotify->OnDeviceLost();
    }

    for (UINT n = 0; n < m_backBufferCount; n++)
    {
        m_commandAllocators[n].Reset();
    }

    m_commandQueue.Reset();
    m_commandList.Reset();
    m_fence.Reset();
    m_rtvDescriptorHeap.Reset();
    m_dsvDescriptorHeap.Reset();
    m_d3dDevice.Reset();
    m_dxgiFactory.Reset();

#ifdef _DEBUG
    {
        ComPtr<IDXGIDebug1> dxgiDebug;
        if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiDebug))))
        {
            dxgiDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_IGNORE_INTERNAL));
        }
    }
#endif

    CreateDeviceResources();

    if (m_deviceNotify)
    {
        m_deviceNotify->OnDeviceRestored();
    }
}

// Prepare the command list and render target for rendering.
void DeviceResources::Prepare()
{
    // Reset command list and allocator.
    ThrowIfFailed(m_commandAllocators[m_backBufferIndex]->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_backBufferIndex].Get(), nullptr));
}

// Present the contents of the swap chain to the screen.
void DeviceResources::SubmitWork(HANDLE inputWaitFence, uint64_t inputWaitFenceValue, HANDLE* outputWaitFence, uint64_t* outputWaitFenceValue)
{
    // Send the command list off to the GPU for processing.
    ThrowIfFailed(m_commandList->Close());
    if (inputWaitFence)
    {
        ComPtr<ID3D12Fence> wait_fence;
        ThrowIfFailed(m_d3dDevice->OpenSharedHandle(inputWaitFence, IID_PPV_ARGS(&wait_fence)));
        m_commandQueue->Wait(wait_fence.Get(), inputWaitFenceValue);
    }
    
    m_commandQueue->ExecuteCommandLists(1, CommandListCast(m_commandList.GetAddressOf()));

    MoveToNextFrame(outputWaitFence, outputWaitFenceValue);
}

// Wait for pending GPU work to complete.
void DeviceResources::WaitForGpu() noexcept
{
    if (m_commandQueue && m_fence && m_fenceEvent.IsValid())
    {
        // Schedule a Signal command in the GPU queue.
        UINT64 fenceValue = m_fenceValues[m_backBufferIndex];
        if (SUCCEEDED(m_commandQueue->Signal(m_fence.Get(), fenceValue)))
        {
            // Wait until the Signal has been processed.
            if (SUCCEEDED(m_fence->SetEventOnCompletion(fenceValue, m_fenceEvent.Get())))
            {
                WaitForSingleObjectEx(m_fenceEvent.Get(), INFINITE, FALSE);

                // Increment the fence value for the current frame.
                m_fenceValues[m_backBufferIndex]++;
            }
        }
    }
}

// Prepare to render the next frame.
void DeviceResources::MoveToNextFrame(HANDLE* outputWaitFence, uint64_t* outputWaitFenceValue)
{
    // Schedule a Signal command in the queue.
    const UINT64 currentFenceValue = m_fenceValues[m_backBufferIndex];
    if (outputWaitFenceValue)
        *outputWaitFenceValue = currentFenceValue;
    if (outputWaitFence)
        *outputWaitFence = m_shared_fence_handle;

    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));

    // Update the back buffer index.
    m_backBufferIndex = (m_backBufferIndex + 1) % m_backBufferCount;

    // If the next frame is not ready to be rendered yet, wait until it is ready.
    if (m_fence->GetCompletedValue() < m_fenceValues[m_backBufferIndex])
    {
        ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_backBufferIndex], m_fenceEvent.Get()));
        WaitForSingleObjectEx(m_fenceEvent.Get(), INFINITE, FALSE);
    }

    // Set the fence value for the next frame.
    m_fenceValues[m_backBufferIndex] = currentFenceValue + 1;
}

// This method acquires the first available hardware adapter that supports Direct3D 12.
// If no such adapter can be found, try WARP. Otherwise throw an exception.
void DeviceResources::GetAdapter(IDXGIAdapter1** ppAdapter)
{
    *ppAdapter = nullptr;

    ComPtr<IDXGIAdapter1> adapter;

#if defined(__dxgi1_6_h__) && defined(NTDDI_WIN10_RS4)
    ComPtr<IDXGIFactory6> factory6;
    HRESULT hr = m_dxgiFactory.As(&factory6);
    if (SUCCEEDED(hr))
    {
        for (UINT adapterIndex = 0;
            SUCCEEDED(factory6->EnumAdapterByGpuPreference(
                adapterIndex,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf())));
            adapterIndex++)
        {
            DXGI_ADAPTER_DESC1 desc;
            ThrowIfFailed(adapter->GetDesc1(&desc));

            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                // Don't select the Basic Render Driver adapter.
                continue;
            }

            // Check to see if the adapter supports Direct3D 12, but don't create the actual device yet.
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), m_d3dMinFeatureLevel, _uuidof(ID3D12Device), nullptr)))
            {
#ifdef _DEBUG
                wchar_t buff[256] = {};
                swprintf_s(buff, L"Direct3D Adapter (%u): VID:%04X, PID:%04X - %ls\n", adapterIndex, desc.VendorId, desc.DeviceId, desc.Description);
                OutputDebugStringW(buff);
#endif
                break;
            }
        }
    }
#endif
    if (!adapter)
    {
        for (UINT adapterIndex = 0;
            SUCCEEDED(m_dxgiFactory->EnumAdapters1(
                adapterIndex,
                adapter.ReleaseAndGetAddressOf()));
            ++adapterIndex)
        {
            DXGI_ADAPTER_DESC1 desc;
            ThrowIfFailed(adapter->GetDesc1(&desc));

            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                // Don't select the Basic Render Driver adapter.
                continue;
            }

            // Check to see if the adapter supports Direct3D 12, but don't create the actual device yet.
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), m_d3dMinFeatureLevel, _uuidof(ID3D12Device), nullptr)))
            {
#ifdef _DEBUG
                wchar_t buff[256] = {};
                swprintf_s(buff, L"Direct3D Adapter (%u): VID:%04X, PID:%04X - %ls\n", adapterIndex, desc.VendorId, desc.DeviceId, desc.Description);
                OutputDebugStringW(buff);
#endif
                break;
            }
        }
    }

#if !defined(NDEBUG)
    if (!adapter)
    {
        // Try WARP12 instead
        if (FAILED(m_dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf()))))
        {
            throw std::exception("WARP12 not available. Enable the 'Graphics Tools' optional feature");
        }

        OutputDebugStringA("Direct3D Adapter - WARP12\n");
    }
#endif

    if (!adapter)
    {
        throw std::exception("No Direct3D 12 device found");
    }

    *ppAdapter = adapter.Detach();
}

