#pragma once

#include <cstdint>

// This is a struct used to communicate launch parameters which are constant
// for all threads in a given optixLaunch call. 
struct Params
{
    uint32_t* image;
    unsigned int  image_width;
    unsigned int  image_height;
    float3   cam_eye;
    float3   cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

// These structs represent the data blocks of our SBT records
struct RayGenData {
    // No data needed
};
struct HitGroupData {
    // No data needed
};
struct MissData {
    float3 bg_color;
};