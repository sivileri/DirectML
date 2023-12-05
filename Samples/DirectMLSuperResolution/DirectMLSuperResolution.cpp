//--------------------------------------------------------------------------------------
// DirectMLSuperResolution.cpp
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. Copyright (C) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

#include "pch.h"

#include "DirectMLSuperResolution.h"

#include "ATGColors.h"
#include "ControllerFont.h"
#include "FindMedia.h"
#include "ReadData.h"
#include "Float16Compressor.h"

using namespace DirectX;

using Microsoft::WRL::ComPtr;

#pragma warning(disable : 4238)

namespace
{
    struct Vertex
    {
        XMFLOAT4 position;
        XMFLOAT2 texcoord;
    };

    struct ImageLayoutCB
    {
        UINT Height;
        UINT Width;
        bool UseNhwc;
    };

    // Divide and round up
    static UINT DivUp(UINT a, UINT b)
    {
        return (a + b - 1) / b;
    }
}

Upscaler::Upscaler()
    : m_tensorLayout(TensorLayout::Default),
    m_srcTextureWidth(0),
    m_srcTextureHeight(0),
    m_VideoTextureHandle(NULL),
    m_indexBufferView({}),
    m_vertexBufferView({})
{
    // Renders only 2D, so no need for a depth buffer.
    m_deviceResources = std::make_unique<DX::DeviceResources>(DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN,
        c_backBufferCount, D3D_FEATURE_LEVEL_11_0, 0);
    m_deviceResources->RegisterDeviceNotify(this);
}

Upscaler::~Upscaler()
{
    if (m_deviceResources)
    {
        m_deviceResources->WaitForGpu();
    }
}

// Initialize the Direct3D resources required to run.
void Upscaler::Initialize(int width, int height, HANDLE* pSrcSharedHandle, HANDLE* pDstSharedHandle)
{
    m_srcTextureWidth = width;
    m_srcTextureHeight = height;
    m_deviceResources->CreateDeviceResources();
    CreateDeviceDependentResources();
    *pSrcSharedHandle = m_VideoTextureHandle;
    *pDstSharedHandle = m_finalResultTextureHandle;
}

#pragma region Frame Update
#pragma region Frame Render
// Draws the scene.
void Upscaler::Render(HANDLE inputWaitFence, uint64_t inputWaitFenceValue, HANDLE* outputWaitFence, uint64_t* outputWaitFenceValue)
{
    // Prepare the command list to render a new frame.
    m_deviceResources->Prepare();
    
    auto commandList = m_deviceResources->GetCommandList();

    // Convert image to tensor format (original texture -> model input)
    {
        ID3D12DescriptorHeap* pHeaps[] = { m_SRVDescriptorHeap->Heap() };
        commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

        commandList->SetComputeRootSignature(m_computeRootSignature.Get());

        ImageLayoutCB imageLayoutCB = {};
        imageLayoutCB.Height = m_srcTextureHeight;
        imageLayoutCB.Width = m_srcTextureWidth;
        imageLayoutCB.UseNhwc = (m_tensorLayout == TensorLayout::NHWC);

        commandList->SetComputeRoot32BitConstants(e_crpIdxCB, 3, &imageLayoutCB, 0);
        commandList->SetComputeRootDescriptorTable(e_crpIdxSRV, m_SRVDescriptorHeap->GetGpuHandle(e_descTexture));
        commandList->SetComputeRootDescriptorTable(e_crpIdxUAV, m_SRVDescriptorHeap->GetGpuHandle(e_descModelInput));

        commandList->SetPipelineState(m_computePSO.Get());
        commandList->Dispatch(DivUp(m_srcTextureWidth, 32), DivUp(m_srcTextureHeight, 16), 1);

        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
    }

    // Run the DirectML operations (model input -> model output)
    {
        ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
        commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);
        m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlGraph.Get(), m_dmlBindingTable.Get());
        // UAV barrier handled below
    }

    // Render either the DML result to a texture
    {
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_finalResultTexture.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET),
            CD3DX12_RESOURCE_BARRIER::Transition(m_modelOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            CD3DX12_RESOURCE_BARRIER::UAV(nullptr)    
        };

        commandList->ResourceBarrier(_countof(barriers), barriers);

        auto rtv = m_RTVDescriptorHeap->GetCpuHandle(e_descFinalResultTextureRtv);
        commandList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        // Use linear clear color for gamma-correct rendering.
        commandList->ClearRenderTargetView(rtv, Colors::Black, 0, nullptr);
            
        D3D12_VIEWPORT texViewport = {};
        D3D12_RECT texScissor = {};
        // TODO: If not shader hardcoded deps for 2x upscale, add m_dstTextureWidth/Heigth configurable on exported functions
        texViewport.Height = static_cast<FLOAT>(texScissor.bottom = m_srcTextureHeight * 2);
        texViewport.Width = static_cast<FLOAT>(texScissor.right = m_srcTextureWidth * 2);
            
        commandList->RSSetViewports(1, &texViewport);
        commandList->RSSetScissorRects(1, &texScissor);

        auto heap = m_SRVDescriptorHeap->Heap();

        // Convert output tensor back to image (model output -> final result texture)
        commandList->SetGraphicsRootSignature(m_tensorRenderRootSignature.Get());
        commandList->SetPipelineState(m_tensorRenderPipelineState.Get());
        commandList->SetDescriptorHeaps(1, &heap);

        ImageLayoutCB imageLayoutCB = {};
        imageLayoutCB.Height = m_srcTextureHeight * 2;
        imageLayoutCB.Width = m_srcTextureWidth * 2;
        imageLayoutCB.UseNhwc = (m_tensorLayout == TensorLayout::NHWC);

        commandList->SetGraphicsRoot32BitConstants(e_rrpIdxCB, 3, &imageLayoutCB, 0);
        commandList->SetGraphicsRootDescriptorTable(e_rrpIdxSRV, m_SRVDescriptorHeap->GetGpuHandle(e_descModelOutput));

        // Set necessary state.
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
        commandList->IASetIndexBuffer(&m_indexBufferView);

        // Draw quad.
        commandList->DrawIndexedInstanced(6, 1, 0, 0, 0);
    }

    m_deviceResources->SubmitWork(inputWaitFence, inputWaitFenceValue, outputWaitFence, outputWaitFenceValue);
}
#pragma endregion


#pragma region Direct3D Resources
// These are the resources that depend on the device.
void Upscaler::CreateDeviceDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Create descriptor heaps.
    {
        m_SRVDescriptorHeap = std::make_unique<DescriptorHeap>(
            device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            e_srvDescCount);

        m_RTVDescriptorHeap = std::make_unique<DescriptorHeap>(
            device,
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            e_rtvDescCount);

    }

    CreateTextureResources();
    CreateDirectMLResources();
    InitializeDirectMLResources();
}

void Upscaler::CreateTextureResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Create vertex buffer for full screen texture render.
    {
        static const Vertex s_vertexData[4] =
        {
            { { -1.f, -1.f, 1.f, 1.0f },{ 0.f, 1.f } },
            { { 1.f, -1.f, 1.f, 1.0f },{ 1.f, 1.f } },
            { { 1.f,  1.f, 1.f, 1.0f },{ 1.f, 0.f } },
            { { -1.f,  1.f, 1.f, 1.0f },{ 0.f, 0.f } },
        };

        // Note: using upload heaps to transfer static data like vert buffers is not 
        // recommended. Every time the GPU needs it, the upload heap will be marshalled 
        // over. Please read up on Default Heap usage. An upload heap is used here for 
        // code simplicity and because there are very few verts to actually transfer.
        DX::ThrowIfFailed(
            device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(sizeof(s_vertexData)),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(m_vertexBuffer.ReleaseAndGetAddressOf())));

        // Copy the quad data to the vertex buffer.
        UINT8* pVertexDataBegin;
        CD3DX12_RANGE readRange(0, 0);		// We do not intend to read from this resource on the CPU.
        DX::ThrowIfFailed(
            m_vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
        memcpy(pVertexDataBegin, s_vertexData, sizeof(s_vertexData));
        m_vertexBuffer->Unmap(0, nullptr);

        // Initialize the vertex buffer view.
        m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
        m_vertexBufferView.StrideInBytes = sizeof(Vertex);
        m_vertexBufferView.SizeInBytes = sizeof(s_vertexData);
    }

    // Create index buffer.
    {
        static const uint16_t s_indexData[6] =
        {
            3,1,0,
            2,1,3,
        };

        // See note above
        DX::ThrowIfFailed(
            device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(sizeof(s_indexData)),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(m_indexBuffer.ReleaseAndGetAddressOf())));

        // Copy the data to the index buffer.
        UINT8* pVertexDataBegin;
        CD3DX12_RANGE readRange(0, 0);		// We do not intend to read from this resource on the CPU.
        DX::ThrowIfFailed(
            m_indexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
        memcpy(pVertexDataBegin, s_indexData, sizeof(s_indexData));
        m_indexBuffer->Unmap(0, nullptr);

        // Initialize the index buffer view.
        m_indexBufferView.BufferLocation = m_indexBuffer->GetGPUVirtualAddress();
        m_indexBufferView.Format = DXGI_FORMAT_R16_UINT;
        m_indexBufferView.SizeInBytes = sizeof(s_indexData);
    }

    {
        // Create texture to receive video frames.
        CD3DX12_RESOURCE_DESC desc(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            0,
            m_srcTextureWidth,
            m_srcTextureHeight,
            1,
            1,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            1,
            0,
            D3D12_TEXTURE_LAYOUT_UNKNOWN,
            D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);

        CD3DX12_HEAP_PROPERTIES defaultHeapProperties(D3D12_HEAP_TYPE_DEFAULT);

        DX::ThrowIfFailed(
            device->CreateCommittedResource(
                &defaultHeapProperties,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                nullptr,
                IID_PPV_ARGS(m_videoTexture.ReleaseAndGetAddressOf())));

        CreateShaderResourceView(device, m_videoTexture.Get(), m_SRVDescriptorHeap->GetCpuHandle(e_descTexture));

        DX::ThrowIfFailed(
            device->CreateSharedHandle(
                m_videoTexture.Get(),
                nullptr,
                GENERIC_ALL,
                nullptr,
                &m_VideoTextureHandle));
    }
}

void Upscaler::CreateWeightTensors(
    WeightMapType& weights,
    const char* convLayerName,
    const char* scaleLayerName,
    const char* shiftLayerName,
    dml::Span<const uint32_t> filterSizes,
    DirectX::ResourceUploadBatch& uploadBatch,
    _Out_writes_(1) ID3D12Resource** filterWeightResourceOut,
    _Out_writes_opt_(1) ID3D12Resource** biasWeightResourceOut)
{
    // There are two types of weights for the convolutions: The convolution filters themselves, and scale/shift
    // weights used to normalize and bias the results. The final layer doesn't use scale and shift weights, so
    // these are optional.

    bool useScaleShift = true;
    if (scaleLayerName == nullptr)
    {
        assert(shiftLayerName == nullptr);
        useScaleShift = false;
    }
    
    CreateWeightResource(filterSizes.data(), filterWeightResourceOut);
    if (useScaleShift)
    {
        uint32_t biasSizes[] = { 1, filterSizes[0], 1, 1 };	// One bias per output channel
        CreateWeightResource(biasSizes, biasWeightResourceOut);

        // The scale weights will be premultiplied into the filter weights, so they don't need
        // a separate resource.
    }
    else
    {
        if (biasWeightResourceOut)
            biasWeightResourceOut = nullptr;
    }

    // Convert weight values to FP16
    WeightsType filterWeights = weights[convLayerName];
    WeightsType scaleWeights, shiftWeights;
    if (useScaleShift)
    {
        scaleWeights = weights[scaleLayerName];
        shiftWeights = weights[shiftLayerName];
    }

    std::vector<uint16_t> filterWeightsFP16;
    std::vector<uint16_t> biasWeightsFP16;

    const uint32_t N = filterSizes[0];
    const uint32_t C = filterSizes[1];
    const uint32_t H = filterSizes[2];
    const uint32_t W = filterSizes[3];

    for (uint32_t n = 0; n < N; n++)
    {
        switch (m_tensorLayout)
        {
        case TensorLayout::NHWC:
            // We need to convert the weights from NCHW to NHWC.
            for (uint32_t h = 0; h < H; h++)
                for (uint32_t w = 0; w < W; w++)
                    for (uint32_t c = 0; c < C; c++)
                    {
                        // Apply the scale weight now so we don't need a normalization layer
                        uint32_t idx = w + h * W + c * H*W + n * C*H*W;
                        float scaledWeight = useScaleShift ?
                            filterWeights[idx] * scaleWeights[n] :
                            filterWeights[idx];
                        filterWeightsFP16.push_back(Float16Compressor::compress(scaledWeight));
                    }
            break;

        default:
            // Weights are already in the right order
            for (uint32_t i = 0; i < C*H*W; i++)
            {
                // Apply the scale weight now so we don't need a normalization layer
                uint32_t idx = n * C*H*W + i;
                float scaledWeight = useScaleShift ?
                    filterWeights[idx] * scaleWeights[n] :
                    filterWeights[idx];
                filterWeightsFP16.push_back(Float16Compressor::compress(scaledWeight));
            }
        }

        if (useScaleShift)
        {
            // Technically this is initialBias*scale+shift, but the initial bias is 0
            biasWeightsFP16.push_back(Float16Compressor::compress(shiftWeights[n]));
        }
    }

    // Upload to the GPU
    D3D12_SUBRESOURCE_DATA weightsData = {};
    weightsData.pData = filterWeightsFP16.data();
    uploadBatch.Upload(*filterWeightResourceOut, 0, &weightsData, 1);

    if (useScaleShift)
    {
        weightsData.pData = biasWeightsFP16.data();
        uploadBatch.Upload(*biasWeightResourceOut, 0, &weightsData, 1);
    }
}

void Upscaler::GetStrides(
    _In_reads_(4) const uint32_t* sizes,
    TensorLayout layout,
    _Out_writes_(4) uint32_t* stridesOut
)
{
    switch (layout)
    {
    case TensorLayout::NHWC:
        stridesOut[0] = sizes[1] * sizes[2] * sizes[3];
        stridesOut[1] = 1;
        stridesOut[2] = sizes[1] * sizes[3];
        stridesOut[3] = sizes[1];
        break;

    default:
        stridesOut[0] = sizes[1] * sizes[2] * sizes[3];
        stridesOut[1] = sizes[2] * sizes[3];
        stridesOut[2] = sizes[3];
        stridesOut[3] = 1;
    }
}


void Upscaler::CreateWeightResource(
    _In_reads_(4) const uint32_t* tensorSizes,
    _Out_writes_(1) ID3D12Resource** d3dResourceOut)
{
    uint32_t strides[4];
    GetStrides(tensorSizes, m_tensorLayout, strides);
    uint64_t bufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, tensorSizes, strides);

    D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(d3dResourceOut)
    ));
}

void Upscaler::OnDeviceLost()
{
    m_tensorRenderPipelineState.Reset();
    m_tensorRenderRootSignature.Reset();
    m_videoTexture.Reset();
    m_finalResultTexture.Reset();
    m_indexBuffer.Reset();
    m_vertexBuffer.Reset();

    m_SRVDescriptorHeap.reset();
    m_RTVDescriptorHeap.reset();

    m_computePSO.Reset();
    m_computeRootSignature.Reset();

    m_dmlDevice.Reset();
    m_dmlCommandRecorder.Reset();

    m_modelInput.Reset();
    m_modelOutput.Reset();

    m_dmlOpInitializer.Reset();
    m_dmlGraph.Reset();
    m_modelTemporaryResource.Reset();
    m_modelPersistentResource.Reset();

    for (int i = 0; i < c_numConvLayers; i++)
    {
        m_modelConvFilterWeights[i].Reset();
        m_modelConvBiasWeights[i].Reset();
    }
    m_dmlDescriptorHeap.reset();
}

void Upscaler::OnDeviceRestored()
{
    CreateDeviceDependentResources();
}
#pragma endregion

void CreateUpscaler(int src_width, int src_height, void** pUPScaler, HANDLE* pSrcSharedResource, HANDLE* pDstSharedResource)
{
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
    Upscaler* upscaler = new Upscaler();
    upscaler->Initialize(src_width, src_height, pSrcSharedResource, pDstSharedResource);
    *pUPScaler = (void*) upscaler;
}

void DeleteUpscaler(void *pUPScaler)
{
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
    delete pUPScaler;
}

void RenderUpscale(void* pUPScaler, HANDLE inputWaitFence, uint64_t inputWaitFenceValue, HANDLE *outputWaitFence, uint64_t* outputWaitFenceValue)
{
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
    Upscaler* upscaler = (Upscaler*) pUPScaler;
    upscaler->Render(inputWaitFence, inputWaitFenceValue, outputWaitFence, outputWaitFenceValue);
}

uint32_t GetMaxBackBuffers()
{
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
    return Upscaler::c_backBufferCount;
}