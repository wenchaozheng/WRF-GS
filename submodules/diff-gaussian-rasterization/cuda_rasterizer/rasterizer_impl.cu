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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
// Find the position of the highest valid bit, with bits starting at 1 (e.g. if bit 18 is 1 and all higher bits are 0, return 18)
// (bit 18, i.e. 2^17, is 1 more than the exponent so it's called next-highest???)
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2; // binary search
		if (n >> msb) // Right shift msb bit, if there is no valid number before the last msb bit of n, then msb -= step that is, to the low bit to find, otherwise to the high bit to lookup
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void markAllVisible(int P,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	present[idx] = true;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,                   // Input: 3D Gaus number
	const float2* points_xy, // Input: image space Pixel coordinates
	const float* depths,     // Input: camera coordinates Depth
	const uint32_t* offsets, // Input: number of tile points covered (cumulative value)
	uint64_t* gaussian_keys_unsorted,   // Output: (unsorted) instance key
	uint32_t* gaussian_values_unsorted, // Output: (unsorted) instance Gaus idx
	int* radii,              // Input: image space Radius
	dim3 grid /*Input: dimension shape of the tile grid (number of grids)*/)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0) // Only the Gaus that were not skipped in the preprocess will be involved in the processing, generating key
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// Each Gaus sets aside a number of empty spaces covering the tile for its key
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// Calculate the range box corresponding to Gaus in image space based on the center and radius (denoted by the X and Y ordinal numbers of the tile on which the two diagonal vertices of the box fall, so traversing this result below is equivalent to traversing the tile covered by the box directly by ordinal number)
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		// Each Gaus has a 64-bit key for the number of tiles it covers, i.e. each Gaus instance has 1 key.
		// The upper 32 bits of each key are the one-dimensional number of the tile (first in line, then in columns), and the lower 32 bits are the depth of the camera coordinates, so that the instances are sorted by tile first, and then by depth.
		// The value of the key is the idx of the Gaus, which is used to reverse lookup the corresponding Gaus from the instance.
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	bool* present)
{
	markAllVisible << <(P + 255) / 256, 256 >> > (
		P,
		present);
}



/**
 * Create a new GeometryState object to hold the data of P 3D points
 * Inner layer: all points of the same kind of data under a pointer (continuous); outer layer: different data occupy memory blocks also continuous
 */
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

/**
 * Create a new ImageState object to hold the data of N pixels
 * Inner layer: all pixels of the same kind of data are placed under one pointer (continuous); outer layer: different data occupy memory blocks that are also continuous
 */
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

/**
 * Create a new BinningState object that holds data for num_rendered actual equivalent rendered points.
 * Inner: same data for all points under one pointer (contiguous); outer: different data occupies also contiguous memory blocks
 */
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}



// Forward rendering procedure for differentiable rasterization
// of Gaussians.
// Only out_color (which renders the resulting image) and radii (the gaus radius, which is used to cull gaus, determine which gaus' gradients to update, etc., i.e., to control the densify process) are real outputs; the rest are inputs or contexts
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P/*3D Gaus number*/, int D/*active_sh_degree_，only used by preprocess*/, int M/*Sum of dc and rest harmonics (1+15)*/,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* cam_pos,
	const bool prefiltered,
	float* out_color,
	int* radii)
// int* radii_x)
{
	// The space required for P 3D points (number of chars, i.e. bytes) is first calculated, then space is allocated via the resize_ interface of the incoming torch Tensor, and finally the location of each member pointer of State is divided on this space
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	// If no radii then use the built-in (but in practice always get a tensor passed in from outside)
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// Delineate the tiles: the number of tiles in each dimension needs to be rounded up to 1 - no tailing.
	// [Camera model] Perspective camera's tiles are drawn directly as squares.
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// Calculate the space needed for width * height pixels (number of chars, i.e. bytes), then allocate the space via the incoming torch Tensor's resize_ interface, and finally divide the space between State's member pointers.
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// The Gaussian rasterizer can only compute RGB colors on its own, the rest of the color space needs to be pre-calculated from outside.
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// Calculate covariance, range frame in 2D image space for each 3D Gaus point, and SH to RGB (calculate color in image space)
	FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		radii,
// radii_x,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// Input: tiles_touched: number of tiles covered by each Gaus point.
	// Output: point_offsets: the cumulative number of times all Gaus points have covered a tile up to each Gaus point (the notion of “person times”, point times)
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	//num_rendered = Cumulative number of tile points covered by all Gaus up to the last Gaus (total number of equivalent points to be calculated for the full rendering)
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
	// binning存放参与渲染的等效点云
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// For each Gaus, the actual number of times it has participated in rendering (i.e., the number of tiles it covers) is generated as an instance, and the instance is sorted by the number of tiles it is on and the depth of the corresponding Gaus.
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
// radii_x,
		tile_grid
		);

	// Logarithm of the total number of tile grids in base 2, rounded up, e.g. if there are 100 grids, bit=7 (2^6=64,2^7=128)
	// Used to calculate the highest valid bit of the key, provided for cuda sorting (good for performance?).
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// Sort Gaus instances in ascending order, depth first, then tile number (i.e. same tile together, internally by depth).
	// Sorting is done to be able to generate back-indexed ranges, so that each tile can easily locate which instances it contains.
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit/*The lower 32 bits are the depth, the complete comparison; the higher 32 bits of the tile number can be determined based on the total number of tiles to determine the maximum number to determine the comparison of its low bit can be*/);

	// 各个像素的ranges初始化为(0,0)
	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	// Calculate ranges, the number of Gaus instances each tile contains
	// ranges.x: starting point of instance number
	// ranges.y: start of the instance number of the next tile, i.e. the end of the instance number of the current tile + 1.
	// Iterate through each sorted instance in parallel.
	// If it is the first, then ranges.x = 0 for the corresponding tile.
	// If it's a split point (i.e., its previous instance belongs to a previous tile, different from the one it belongs to), then ranges.y of the previous tile = ranges.x of the tile it belongs to = its idx
	// if it is the last one, then ranges.y of its corresponding tile = total number of instances (point times)
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges
			);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* campos,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix, // grad_outputs[0], i.e., d{loss}_d{forward output}[0], is computed automatically by torch, and the following backward is defined as d{forward output}_d{forward input}
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dpx_dt,
	float* dpy_dt)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R); // Get the number of Gaus instances R via ctx context
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// Divide the tiles: the number of tiles in each dimension needs to be rounded up to 1 - no tails!
	// Divide in screen-space
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// The covariance matrix may be pre-calculated from the full input, or it may be calculated from the input scale and rot inside the rasterizer, note that the two cases are handled differently.
	// Generally the second one is used
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		width, height,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		(float3*)dpx_dt,
		(float3*)dpy_dt);
}
