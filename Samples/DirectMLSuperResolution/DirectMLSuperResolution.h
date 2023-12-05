//-------------------------------------------------------------------------------------
// DirectMLSuperResolution.h
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. Copyright (C) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

#pragma once

#include "DeviceResources.h"
#include "LoadWeights.h"

// Force the default NCHW (batch/channels/height/width) tensor format, instead of determining
// this based on the GPU vendor. Setting this may help run on older Nvidia hardware.
#define FORCE_NCHW 0

// Let DirectML manage the data in the weight tensors. This can be faster on some hardware.
#define DML_MANAGED_WEIGHTS 1

enum class TensorLayout
{
    Default,
    NHWC
};

class Upscaler final : public DX::IDeviceNotify
{
public:
    static const UINT c_backBufferCount = 8;

    Upscaler() noexcept(false);
    ~Upscaler();

    void Initialize(int width, int height, HANDLE* pSrcSharedHandle, HANDLE* pDstSharedHandle);
    void Render(HANDLE inputWaitFence, uint64_t inputWaitFenceValue, HANDLE* outputWaitFence, uint64_t* outputWaitFenceValue);

    // IDeviceNotify
    virtual void OnDeviceLost() override;
    virtual void OnDeviceRestored() override;

private:
    void CreateDeviceDependentResources();
    void CreateTextureResources();
    void CreateDirectMLResources();
    void InitializeDirectMLResources();

    void GetStrides(
        _In_reads_(4) const uint32_t* sizes,
        TensorLayout layout,
        _Out_writes_(4) uint32_t* stridesOut
    );
    void CreateWeightTensors(
        WeightMapType& weights,
        const char* convLayerName,
        const char* scaleLayerName,
        const char* shiftLayerName,
        dml::Span<const uint32_t> filterSizes,
        DirectX::ResourceUploadBatch& uploadBatch,
        _Out_writes_(1) ID3D12Resource** filterWeightResourceOut,
        _Out_writes_opt_(1) ID3D12Resource** biasWeightResourceOut);
    void CreateWeightResource(
        _In_reads_(4) const uint32_t* tensorSizes,
        _Out_writes_(1) ID3D12Resource** d3dResourceOut);

    // Device resources
    std::unique_ptr<DX::DeviceResources>            m_deviceResources;

    // DirectXTK objects
    std::unique_ptr<DirectX::DescriptorHeap>        m_SRVDescriptorHeap;
    std::unique_ptr<DirectX::DescriptorHeap>        m_RTVDescriptorHeap;

    // Direct3D 12 objects for rendering texture to screen
    HANDLE                                          m_VideoTextureHandle;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_tensorRenderRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_tensorRenderPipelineState;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_videoTexture;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_finalResultTexture;
    HANDLE                                          m_finalResultTextureHandle;
    uint32_t                                        m_srcTextureHeight;
    uint32_t                                        m_srcTextureWidth;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_vertexBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_indexBuffer;
    D3D12_VERTEX_BUFFER_VIEW                        m_vertexBufferView;
    D3D12_INDEX_BUFFER_VIEW                         m_indexBufferView;

    // Compute objects for converting texture to DML tensor format
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_computePSO;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_computeRootSignature;

    // DirectML objects
    Microsoft::WRL::ComPtr<IDMLDevice>              m_dmlDevice;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder>     m_dmlCommandRecorder;

    TensorLayout                                    m_tensorLayout;

    // Shared Resources
    std::unique_ptr<DirectX::DescriptorHeap>        m_dmlDescriptorHeap;
    
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelOutput;

    // DirectMLX Model Resources
    static const size_t                             c_numConvLayers = 7;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelConvFilterWeights[c_numConvLayers];
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelConvBiasWeights[c_numConvLayers];

    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelPersistentResource;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelTemporaryResource;

    // DirectMLX operations
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>    m_dmlGraph;
    Microsoft::WRL::ComPtr<IDMLBindingTable>        m_dmlBindingTable;
    Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_dmlOpInitializer;

    // DirectX index enums
    enum SrvDescriptors : uint32_t
    {
        e_descTexture,
        e_descModelInput,
        e_descModelOutput,
        e_descFinalResultTextureSrv,
        e_srvDescCount
    };

    enum RtvDescriptors : uint32_t
    {
        e_descFinalResultTextureRtv,
        e_rtvDescCount
    };

    enum ComputeRootParameters : uint32_t
    {
        e_crpIdxCB = 0,
        e_crpIdxSRV,
        e_crpIdxUAV,
        e_crpIdxCount
    };

    enum TensorRenderRootParameters : uint32_t
    {
        e_rrpIdxCB = 0,
        e_rrpIdxSRV,
        e_rrpIdxCount
    };
};

__declspec(dllexport) void __cdecl CreateUpscaler(int src_width, int src_height, void** pUPScaler, HANDLE* pSrcSharedResource, HANDLE* pDstSharedResource);
__declspec(dllexport) void __cdecl DeleteUpscaler(void* pUPScaler);
__declspec(dllexport) void __cdecl RenderUpscale(void* pUPScaler, HANDLE inputWaitFence, uint64_t inputWaitFenceValue, HANDLE* outputWaitFence, uint64_t* outputWaitFenceValue);
__declspec(dllexport) uint32_t __cdecl GetMaxBackBuffers(); /* Number of in-flight operations until overwriting circular buffer */
