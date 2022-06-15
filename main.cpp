#include "rob_optix.h"
#include "launch_params.h"
#include "exceptions.h"
#include "ptx_loader.h"

#include <array>
#include <iostream>

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

// SBT record with an appropriately aligned and sized data block
template<typename T> struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

char olog[2048]; // For error reporting from OptiX creation functions

int main() {
    // Initialize CUDA with a no-op call to the the CUDA runtime API
    CUDA_CHECK( cudaFree(0) );

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("#rob: no CUDA capable devices found!");
    std::cout << "#rob: found " << numDevices << " CUDA devices" << std::endl;

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK( optixInit() );

    // Specify options for this context. We will use the default options.
    OptixDeviceContextOptions options = {};

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    CUcontext cuCtx = 0; // NULL means take the current active context

    OptixDeviceContext context = nullptr;
    OPTIX_CHECK( optixDeviceContextCreate(cuCtx, &options, &context) );

    // Specify options for the build. We use default options for simplicity.
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Triangle build input: simple list of three vertices
    const std::array<float3, 3> vertices = {
        {
            {-0.5f, -0.5f, 0.0f},
            {0.5f, -0.5f, 0.0f},
            {0.0f, 0.5f, 0.0f}
        }
    };

    // Allocate and copy device memory for our input triangle vertices
    const size_t vertices_size = sizeof(float3) * vertices.size();
    CUdeviceptr d_vertices = 0;
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size) );
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(d_vertices),
        vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice) );

    // Populate the build input struct with our triangle data as well as
    // information about the sizes and types of our data
    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = vertices.size();
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    // Query OptiX for the memory requirements for our GAS
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        context, // The device context we are using
        &accel_options,
        &triangle_input, // Describes our geometry
        1,               // Number of build inputs, could have multiple
        &gas_buffer_sizes) );

    // Allocate device memory for the scratch space buffer as well
    // as the GAS itself
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes) );
    CUdeviceptr d_gas_output_buffer;
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer),
        gas_buffer_sizes.outputSizeInBytes) );

    // Now build the GAS
    OptixTraversableHandle gas_handle{ 0 };
    OPTIX_CHECK( optixAccelBuild(
        context,
        0, // CUDA stream
        &accel_options,
        &triangle_input,
        1, // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle, // Output handle to the struct
        nullptr,     // emitted property list
        0) );          // num emitted properties

    // We can now free scratch space used during the build
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)) );

    // Default options for our module.
    OptixModuleCompileOptions module_compile_options = {};

    // Pipeline options must be consistent for all modules used in a
    // single pipeline
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;

    // This option is important to ensure we compile code which is optimal
    // for our scene hierarchy. We use a single GAS – no instancing or
    // multi-level hierarchies
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    // Our device code uses 3 payload registers (r,g,b output value)
    pipeline_compile_options.numPayloadValues = 3;

    // This is the name of the param struct variable in our device code
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t sizeof_olog = sizeof(olog);

    size_t      inputSize = 0;
    const char* input = rob::getInputData("vec.cu", inputSize);

    OptixModule module = nullptr; // The output module
    OPTIX_CHECK( optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        input,
        inputSize,
        olog,
        &sizeof_olog,
        &module) );

    return 0;
}