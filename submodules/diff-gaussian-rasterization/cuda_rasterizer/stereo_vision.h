/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include "config.h"
#include "stdio.h"
#include "auxiliary.h"

#ifndef BLOCK_X
    #define BLOCK_X 16
#endif

#ifndef BLOCK_Y
    #define BLOCK_Y 16
#endif

#ifndef BLOCK_SIZE
    #define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#endif

#ifndef NUM_WARPS
    #define NUM_WARPS (BLOCK_SIZE/32)
#endif

__forceinline__ __device__ float3 reproject_depth_pinhole(
    const int u,
    const int v,
    const float depth,
    const float fx,
    const float fy,
    const float cx,
    const float cy)
{
    float3 pt;
    pt.x = (u - cx) * depth / fx;
    pt.y = (v - cy) * depth / fy;
    pt.z = depth;
    return pt;
}
