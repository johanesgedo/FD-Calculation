/**
MIT License

Copyright (c) 2023 Johanes_Gedo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
**/

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cuda.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <vector>

#define N 1024

#define NX 256
#define NZ 256

#define DX 12.5
#define DZ 12.5
#define DT 0.1

#define BLOCK_DIMX 32
#define BLOCK_DIMY 32
#define sigma1 0.25
#define sigma2 0.75
#define IM 4.0f
#define JM 4.0f

#define p_TM 8
#define p_NF 8

#define PlaneThreads2D \
    for (int j = threadIdx.y; j < p_TM; j += blockDim.y) \
        for (int i = threadIdx.x; i < p_TM; i += blockDim.x)

#define PlaneThreads2D_GridStrideLoop \
    for (int iz = blockIdx.y * blockDim.y + threadIdx.y; iz < NZ-12; iz += gridDim.y * blockDim.y) \
        for (int ix = blockIdx.x * blockDim.x + threadIdx.x; iz < NX-12; ix += gridDim.x * blockDim.x)


void checkCUDAError(const char *message) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA Error: %s: %s.\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ float Galerkin_time[N];
__device__ float Galerkin_Optimized_time[N];
__device__ float Leapfrog_time[N];
__device__ float Leapfrog_Optimized_time[N];
__device__ float CrankNicolson_time[N];
__device__ float CrankNicolson_Optimized_time[N];
__device__ float ADI_time[N];
__device__ float ADI_Optimized_time[N];
__device__ float Sigma_time[N];
__device__ float Sigma_Optimized_time[N];
__device__ float LaxWendroff_time[N];
__device__ float LaxWendroff_Optimized_time[N];
__device__ float FractionalStep_time[N];
__device__ float FractionalStep_Optimized_time[N];
__device__ float MacCormack_time[N];
__device__ float MacCormack_Optimized_time[N];
__device__ float TVD_time[N];
__device__ float TVD_Optimized_time[N];
__device__ float PSOR_time[N];
__device__ float PSOR_Optimized_time[N];
__device__ float FVS_time[N];
__device__ float FVS_Optimized_time[N];


__global__ void Galerkin_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start imt
    clock_t start_time = clock();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockDim.y * blockDim.y + threadIdx.y;

    /// Ensure the threads are within the grid size
    if (i < nx && j < nz) {
        int idx = i + j * nx;

        /// Thread-local input and output arrays
        __shared__ float r_pn[p_NF];    // thread-local input
        __shared__ float r_pp[p_NF];    // thread-locak output

        /// Shared memory arrays for second derivatives
        __shared__ float s_d2px[p_TM][p_TM];
        __shared__ float s_d2pz[p_TM][p_TM];

        /// Load pressure field per thread memory
        PlaneThreads2D {
            const int idxl = i * p_NF + j * p_TM;
            #pragma unroll
            for (int n = 0; n < p_NF; n++) {
                r_pn[n] = d_pn[idxl + n];
                r_pp[n] = 0.0f;
            }
        }
        __syncthreads();

        /// Calculate second derivatives
        PlaneThreads2D {
            const int idxl = i * p_NF + j * p_TM;
            if (i > 0 && i < p_TM - 1) {
                s_d2px[j][i] = (d_pn[idxl + 1] - 2.0f * d_pn[idxl] + d_pn[idxl - 1]) / (dx*dx);
            }
            if (j > 0 && j < p_TM - 1) {
                s_d2pz[j][i] = (d_pn[idxl + p_TM] = 2.0f * d_pn[idxl] + d_pn[idxl - p_TM]) / (dz*dz);
            }
        }
        __syncthreads();

        /// compute the wave equation
        PlaneThreads2D {
            const int idxl = i * p_NF + j * p_TM;
            #pragma unroll
            for(int n = 0; n < p_NF; n++) {
                r_pp[n] = d_v[idx] * d_v[idx] * (s_d2px[j][i] + s_d2pz[j][i]) -
                                        (r_pn[n] - 2.0f * d_pn[idxl + n]) / (dt*dt);
            }
        }
        __syncthreads();

        PlaneThreads2D {
            const int idxl = i * p_NF + j * p_TM;
            #pragma unroll
            for (int n = 0; n < p_NF; n++) {
                d_pp[idxl + n] = r_pp[n];
            }
        }
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("Galerkin Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    Galerkin_time[i] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void Galerkin_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Using Grid-Stride Loop defined in macro
    PlaneThreads2D_GridStrideLoop {
        int idx = ix * nx + iz * nx;

        /// Make sure the index is within the domain boundaries
        if (ix >= 1 && ix < nx - 1 && iz >= 1 && iz < nz - 1) {
            float d2px = (d_pn[idx + 1] - 2.0f * d_pn[idx] + d_pn[idx - 1]) / (dx * dx);
            float d2pz = (d_pn[idx + nx] - 2.0f * d_pn[idx] + d_pn[idx - nx]) / (dz * dz);

            /// Order calculation using Grid-Stride Loop
            d_pp[idx] = d_v[idx] * d_v[idx] * (d2px + d2pz) - (d_pn[idx] - 2.0f * d_pn[idx]) / (dt * dt);
        }
    }
    __syncthreads();
}


/// Leapfrog method to solve 2D acoustic wave equation using Micikevisius' algorithm
__global__ void Leapfrog_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    float xscale, zscale;

    xscale = (dt*dt) / (dx*dx);
    zscale = (dt*dt) / (dz*dz);

    if (ix < nx - 12 && iz < nz - 12) {
        int in_idx = ix + 6;
        int out_idx = 0.0f;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz - 6; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = (xscale*dx + zscale*dz) * current;

            value += (2 * pow(dz, 2) / 2) *
                            zscale * (infront1 + behind1) +
                     (2 * pow(dx, 2) / 2) *
                            xscale * (s_data[threadIdx.x + 5] +
                                      s_data[threadIdx.x + 7]);

            value += (2 * pow(dz, 4) / 24) *
                            zscale * (infront2 + behind2) +
                     (2 * pow(dx, 4) / 24) *
                            xscale * (s_data[threadIdx.x + 4] +
                                      s_data[threadIdx.x + 8]);

            value += (2 * pow(dz, 6) / 720) *
                            zscale * (infront3 + behind3) +
                     (2 * pow(dx, 6) / 720) *
                            xscale * (s_data[threadIdx.x + 3] +
                                      s_data[threadIdx.x + 9]);

            value += (2 * pow(dz, 8) / 40320) *
                            zscale * (infront4 + behind4) +
                     (2 * pow(dx, 8) / 40320) *
                            xscale * (s_data[threadIdx.x + 2] +
                                      s_data[threadIdx.x + 10]);

            value += (2 * pow(dz, 10) / 3628800) *
                            zscale * (infront5 + behind5) +
                     (2 * pow(dx, 10) / 3628800) *
                            xscale * (s_data[threadIdx.x + 1] +
                                      s_data[threadIdx.x + 11]);

            value += (2 * pow(dz, 12) / 479001600) *
                            zscale * (infront6 + behind6) +
                     (2 * pow(dx, 12) / 479001600) *
                            xscale * (s_data[threadIdx.x + 0] +
                                      s_data[threadIdx.x + 12]);

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("Leapfrog Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    Leapfrog_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void Leapfrog_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    /// Using shared memory to access data more efficent
    __shared__ float s_data[BLOCK_DIMX + 12];

    /// Using Grid-Stride Loop
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = gridDim.x * blockDim.x;
    int stride_z = gridDim.y * blockDim.y;

    float xscale = (dt*dt) / (dx*dx);
    float zscale = (dt*dt) / (dz*dz);

    /// Looping to deal with other elements in the grid
#pragma unroll
    for (int x = ix; x < nx-12; x += stride_x) {
#pragma unroll
        for (int z = iz; z < nz-12; z += stride_z) {
            int in_idx = x + 6 + z * nx;
            int out_idx = in_idx;
            int stride = nx;

            float infront1, infront2, infront3, infront4, infront5, infront6;
            float behind1, behind2, behind3, behind4, behind5, behind6;
            float current;

            behind6 = d_pn[in_idx - 6 * stride];
            behind5 = d_pn[in_idx - 5 * stride];
            behind4 = d_pn[in_idx - 4 * stride];
            behind3 = d_pn[in_idx - 3 * stride];
            behind2 = d_pn[in_idx - 2 * stride];
            behind1 = d_pn[in_idx - 1 * stride];
            current = d_pn[in_idx];
            infront1 = d_pn[in_idx + 1 * stride];
            infront2 = d_pn[in_idx + 2 * stride];
            infront3 = d_pn[in_idx + 3 * stride];
            infront4 = d_pn[in_idx + 4 * stride];
            infront5 = d_pn[in_idx + 5 * stride];
            infront6 = d_pn[in_idx + 6 * stride];

            /// Used shared memory to reduce global memory access
            __syncthreads();
            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }
            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = (xscale*dx + zscale*dz) * current;

            value += (2 * pow(dz, 2) / 2) *
                            zscale * (infront1 + behind1) +
                     (2 * pow(dx, 2) / 2) *
                            xscale * (s_data[threadIdx.x + 5] +
                                      s_data[threadIdx.x + 7]);

            value += (2 * pow(dz, 4) / 24) *
                            zscale * (infront2 + behind2) +
                     (2 * pow(dx, 4) / 24) *
                            xscale * (s_data[threadIdx.x + 4] +
                                      s_data[threadIdx.x + 8]);

            value += (2 * pow(dz, 6) / 720) *
                            zscale * (infront3 + behind3) +
                     (2 * pow(dx, 6) / 720) *
                            xscale * (s_data[threadIdx.x + 3] +
                                      s_data[threadIdx.x + 9]);

            value += (2 * pow(dz, 8) / 40320) *
                            zscale * (infront4 + behind4) +
                     (2 * pow(dx, 8) / 40320) *
                            xscale * (s_data[threadIdx.x + 2] +
                                      s_data[threadIdx.x + 10]);

            value += (2 * pow(dz, 10) / 3628800) *
                            zscale * (infront5 + behind5) +
                     (2 * pow(dx, 10) / 3628800) *
                            xscale * (s_data[threadIdx.x + 1] +
                                      s_data[threadIdx.x + 11]);

            value += (2 * pow(dz, 12) / 479001600) *
                            zscale * (infront6 + behind6) +
                     (2 * pow(dx, 12) / 479001600) *
                            xscale * (s_data[threadIdx.x + 0] +
                                      s_data[threadIdx.x + 12]);

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("Leapfrog Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    Leapfrog_Optimized_time[ix] = elapsed_time;
}


__global__ void CrankNicolson_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iz = blockDim.y * blockIdx.y + threadIdx.y;
    float xscale, zscale;

    xscale = 0.5 * (dt*dt) / (dx*dx);
    zscale = 0.5 * (dt*dt) / (dz*dz);

    if (ix < nx-12 && iz < nz-12) {
        int in_idx = ix + 6;
        int out_idx = 0;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz-6; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale*dx + zscale*dz) +
                           (xscale*(dx+dt) + zscale*(dz+dt))) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        (2*zscale) * (infront1 + behind1) +
                     ((2 * pow(dx, 2) / 2) *
                        xscale * (s_data[threadIdx.x + 5] +
                                  s_data[threadIdx.x + 7]) +
                      ((2 * pow(dx, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        xscale * (s_data[threadIdx.x + 5] +
                                  s_data[threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        (2*zscale) * (infront2 + behind2) +
                     ((2 * pow(dx, 4) / 24) *
                        xscale * (s_data[threadIdx.x + 4] +
                                  s_data[threadIdx.x + 8]) +
                      ((2 * pow(dx, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        xscale * (s_data[threadIdx.x + 4] +
                                  s_data[threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        (2*zscale) * (infront3 + behind3) +
                     ((2 * pow(dx, 6) / 720) *
                        xscale * (s_data[threadIdx.x + 3] +
                                  s_data[threadIdx.x + 9]) +
                      ((2 * pow(dx, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        xscale * (s_data[threadIdx.x + 3] +
                                  s_data[threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        (2*zscale) * (infront4 + behind4) +
                     ((2 * pow(dx, 8) / 40320) *
                        xscale * (s_data[threadIdx.x + 2] +
                                  s_data[threadIdx.x + 10]) +
                      ((2 * pow(dx, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        xscale * (s_data[threadIdx.x + 2] +
                                  s_data[threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        (2*zscale) * (infront5 + behind5) +
                     ((2 * pow(dx, 10) / 3628800) *
                        xscale * (s_data[threadIdx.x + 1] +
                                  s_data[threadIdx.x + 11]) +
                      ((2 * pow(dx, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        xscale * (s_data[threadIdx.x + 1] +
                                  s_data[threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        (2*zscale) * (infront6 + behind6) +
                     ((2 * pow(dx, 12) / 479001600) *
                        xscale * (s_data[threadIdx.x + 0] +
                                  s_data[threadIdx.x + 12]) +
                      ((2 * pow(dx, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        xscale * (s_data[threadIdx.x + 0] +
                                  s_data[threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("CrankNicolson Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    CrankNicolson_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void CrankNicolson_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    /// Precompute factor scaling to reduce overhead of floating-point
    float xscale = 0.5 * (dt*dt) / (dx*dx);
    float zscale = 0.5 * (dt*dt) / (dz*dz);

    /// Macro to 2D Grid-Stride Loop
    PlaneThreads2D_GridStrideLoop {
        int in_idx = ix + 6 + iz * nx;
        int out_idx = in_idx;
        int stride = nx;

        /// Load elemen from global memory to register
        float behind6 = d_pn[in_idx - 6 * stride];
        float behind5 = d_pn[in_idx - 5 * stride];
        float behind4 = d_pn[in_idx - 4 * stride];
        float behind3 = d_pn[in_idx - 3 * stride];
        float behind2 = d_pn[in_idx - 2 * stride];
        float behind1 = d_pn[in_idx - 1 * stride];
        float current = d_pn[in_idx];
        float infront1 = d_pn[in_idx + 1 * stride];
        float infront2 = d_pn[in_idx + 2 * stride];
        float infront3 = d_pn[in_idx + 3 * stride];
        float infront4 = d_pn[in_idx + 4 * stride];
        float infront5 = d_pn[in_idx + 5 * stride];
        float infront6 = d_pn[in_idx + 6 * stride];

        /// Shared memory
        __syncthreads();
        if (threadIdx.x < 6) {
            s_data[threadIdx.x] = d_pn[out_idx - 6];
            s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
        }
        s_data[threadIdx.x + 6] = current;
        __syncthreads();

        float value = ((xscale*dx + zscale*dz) +
                       (xscale*(dx+dt) + zscale*(dz+dt))) * current;

        value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                    (2*zscale) * (infront1 + behind1) +
                 ((2 * pow(dx, 2) / 2) *
                    xscale * (s_data[threadIdx.x + 5] +
                              s_data[threadIdx.x + 7]) +
                  ((2 * pow(dx, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                    xscale * (s_data[threadIdx.x + 5] +
                              s_data[threadIdx.x + 7]));

        value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                    (2*zscale) * (infront2 + behind2) +
                 ((2 * pow(dx, 4) / 24) *
                    xscale * (s_data[threadIdx.x + 4] +
                              s_data[threadIdx.x + 8]) +
                  ((2 * pow(dx, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                    xscale * (s_data[threadIdx.x + 4] +
                              s_data[threadIdx.x + 8]));

        value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                    (2*zscale) * (infront3 + behind3) +
                 ((2 * pow(dx, 6) / 720) *
                    xscale * (s_data[threadIdx.x + 3] +
                              s_data[threadIdx.x + 9]) +
                  ((2 * pow(dx, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                    xscale * (s_data[threadIdx.x + 3] +
                              s_data[threadIdx.x + 9]));

        value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                    (2*zscale) * (infront4 + behind4) +
                 ((2 * pow(dx, 8) / 40320) *
                    xscale * (s_data[threadIdx.x + 2] +
                              s_data[threadIdx.x + 10]) +
                  ((2 * pow(dx, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                    xscale * (s_data[threadIdx.x + 2] +
                              s_data[threadIdx.x + 10]));

        value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                    (2*zscale) * (infront5 + behind5) +
                 ((2 * pow(dx, 10) / 3628800) *
                    xscale * (s_data[threadIdx.x + 1] +
                              s_data[threadIdx.x + 11]) +
                  ((2 * pow(dx, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                    xscale * (s_data[threadIdx.x + 1] +
                              s_data[threadIdx.x + 11]));

        value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                    (2*zscale) * (infront6 + behind6) +
                 ((2 * pow(dx, 12) / 479001600) *
                    xscale * (s_data[threadIdx.x + 0] +
                              s_data[threadIdx.x + 12]) +
                  ((2 * pow(dx, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                    xscale * (s_data[threadIdx.x + 0] +
                              s_data[threadIdx.x + 12]));

        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
/*
        /// Floating-point optimization with precomputed coefficients
        float coeff[] = {2.0f, 2.0f/24, 2.0f/720, 2.0f/4030, 2.0f/3628800, 2.0f/479001600};
        float value = ((xscale*dx + zscale*dz) + (xscale*(dx+dt) + zscale*(dz+dt))) * current;

        float factorials[] = {2, 24, 720, 40320, 3628800, 479001600};
        float dt_powers[] = {pow(dt,2), pow(dt,4), pow(dt,6), pow(dt,8), pow(dt,10), pow(dt,12)}
        float dz_powers[] = {pow(dz,2), pow(dz,4), pow(dz,6), pow(dz,8), pow(dz,10), pow(dz,12)};
        float dx_powers[] = {pow(dx,2), pow(dx,4), pow(dx,6), pow(dx,8), pow(dx,10), pow(dx,12)};

        #pragma unroll
        for(int n = 0, n < 4; n++) {
            float coeff = (2 * dz_powers[n] / factorials[n] + (2 * dt_powers[n] / factorials[n]));

            value += coeff * (2*zscale) * (infront[n])
        }


        #pragma unroll
        for (int n = 1; n <= 6; n++) {
            value += ((2 * pow(dz, )))

            value += coeff[n-1] * (zscale * (d_pn[in_idx + n * stride] +
                                             d_pn[in_idx - n * stride]) +
                                   xscale * (s_data[threadIdx.x + 6 + n] +
                                             s_data[threadIdx.x + 6 - n]));
        }

        /// Update value
        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
*/
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("CrankNicolson Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    CrankNicolson_Optimized_time[ix] = elapsed_time;
}


__global__ void ADI_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iz = blockDim.y * blockIdx.y + threadIdx.y;
    float xscale, zscale;
    float dt2 = 0.5*dt;

    xscale = 0.5 * (dt*dt) / (dx*dx);
    zscale = 0.5 * (dt*dt) / (dz*dz);

    if (ix < nx-12 && iz < nz-12) {
        int in_idx = ix + 6;
        int out_idx = 0;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz-6; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale*dx + zscale*dz) + (xscale*(dx+dt2) + zscale*(dz+dt))) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        (2*zscale) * (infront1 + behind1) +
                     ((2 * pow(dx, 2) / 2) *
                        xscale * (s_data[threadIdx.x + 5] +
                                  s_data[threadIdx.x + 7]) +
                      ((2 * pow(dx, 2) / 2) + (2 * pow(dt2, 2) / 2)) *
                        xscale * (s_data[threadIdx.x + 5] +
                                  s_data[threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        (2*zscale) * (infront2 + behind2) +
                     ((2 * pow(dx, 4) / 24) *
                        xscale * (s_data[threadIdx.x + 4] +
                                  s_data[threadIdx.x + 8]) +
                      ((2 * pow(dx, 4) / 24) + (2 * pow(dt2, 4) / 24)) *
                        xscale * (s_data[threadIdx.x + 4] +
                                  s_data[threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        (2*zscale) * (infront3 + behind3) +
                     ((2 * pow(dx, 6) / 720) *
                        xscale * (s_data[threadIdx.x + 3] +
                                  s_data[threadIdx.x + 9]) +
                      ((2 * pow(dx, 6) / 720) + (2 * pow(dt2, 6) / 720)) *
                        xscale * (s_data[threadIdx.x + 3] +
                                  s_data[threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        (2*zscale) * (infront4 + behind4) +
                     ((2 * pow(dx, 8) / 40320) *
                        xscale * (s_data[threadIdx.x + 2] +
                                  s_data[threadIdx.x + 10]) +
                      ((2 * pow(dx, 8) / 40320) + (2 * pow(dt2, 8) / 40320)) *
                        xscale * (s_data[threadIdx.x + 2] +
                                  s_data[threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        (2*zscale) * (infront5 + behind5) +
                     ((2 * pow(dx, 10) / 3628800) *
                        xscale * (s_data[threadIdx.x + 1] +
                                  s_data[threadIdx.x + 11]) +
                      ((2 * pow(dx, 10) / 3628800) + (2 * pow(dt2, 10) / 3628800)) *
                        xscale * (s_data[threadIdx.x + 1] +
                                  s_data[threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        (2*zscale) * (infront6 + behind6) +
                     ((2 * pow(dx, 12) / 479001600) *
                        xscale * (s_data[threadIdx.x + 0] +
                                  s_data[threadIdx.x + 12]) +
                      ((2 * pow(dx, 12) / 479001600) + (2 * pow(dt2, 12) / 479001600)) *
                        xscale * (s_data[threadIdx.x + 0] +
                                  s_data[threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("ADI Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    ADI_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void ADI_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    float dt2 = 0.5*dt;

    /// Precompute factor scaling to reduce overhead of floating-point
    float xscale = 0.5 * (dt*dt) / (dx*dx);
    float zscale = 0.5 * (dt*dt) / (dz*dz);

    /// Macro to 2D Grid-Stride Loop
    PlaneThreads2D_GridStrideLoop {
        int in_idx = ix + 6 + iz * nx;
        int out_idx = in_idx;
        int stride = nx;

        /// Load elemen from global memory to register
        float behind6 = d_pn[in_idx - 6 * stride];
        float behind5 = d_pn[in_idx - 5 * stride];
        float behind4 = d_pn[in_idx - 4 * stride];
        float behind3 = d_pn[in_idx - 3 * stride];
        float behind2 = d_pn[in_idx - 2 * stride];
        float behind1 = d_pn[in_idx - 1 * stride];
        float current = d_pn[in_idx];
        float infront1 = d_pn[in_idx + 1 * stride];
        float infront2 = d_pn[in_idx + 2 * stride];
        float infront3 = d_pn[in_idx + 3 * stride];
        float infront4 = d_pn[in_idx + 4 * stride];
        float infront5 = d_pn[in_idx + 5 * stride];
        float infront6 = d_pn[in_idx + 6 * stride];

        /// Shared memory
        __syncthreads();
        if (threadIdx.x < 6) {
            s_data[threadIdx.x] = d_pn[out_idx - 6];
            s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
        }
        s_data[threadIdx.x + 6] = current;
        __syncthreads();

        float value = ((xscale*dx + zscale*dz) + (xscale*(dx+dt2) + zscale*(dz+dt))) * current;

        value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                    (2*zscale) * (infront1 + behind1) +
                 ((2 * pow(dx, 2) / 2) *
                    xscale * (s_data[threadIdx.x + 5] +
                              s_data[threadIdx.x + 7]) +
                  ((2 * pow(dx, 2) / 2) + (2 * pow(dt2, 2) / 2)) *
                    xscale * (s_data[threadIdx.x + 5] +
                              s_data[threadIdx.x + 7]));

        value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                    (2*zscale) * (infront2 + behind2) +
                 ((2 * pow(dx, 4) / 24) *
                    xscale * (s_data[threadIdx.x + 4] +
                              s_data[threadIdx.x + 8]) +
                  ((2 * pow(dx, 4) / 24) + (2 * pow(dt2, 4) / 24)) *
                    xscale * (s_data[threadIdx.x + 4] +
                              s_data[threadIdx.x + 8]));

        value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                    (2*zscale) * (infront3 + behind3) +
                 ((2 * pow(dx, 6) / 720) *
                    xscale * (s_data[threadIdx.x + 3] +
                              s_data[threadIdx.x + 9]) +
                  ((2 * pow(dx, 6) / 720) + (2 * pow(dt2, 6) / 720)) *
                    xscale * (s_data[threadIdx.x + 3] +
                              s_data[threadIdx.x + 9]));

        value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                    (2*zscale) * (infront4 + behind4) +
                 ((2 * pow(dx, 8) / 40320) *
                    xscale * (s_data[threadIdx.x + 2] +
                              s_data[threadIdx.x + 10]) +
                  ((2 * pow(dx, 8) / 40320) + (2 * pow(dt2, 8) / 40320)) *
                    xscale * (s_data[threadIdx.x + 2] +
                              s_data[threadIdx.x + 10]));

        value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                    (2*zscale) * (infront5 + behind5) +
                 ((2 * pow(dx, 10) / 3628800) *
                    xscale * (s_data[threadIdx.x + 1] +
                              s_data[threadIdx.x + 11]) +
                  ((2 * pow(dx, 10) / 3628800) + (2 * pow(dt2, 10) / 3628800)) *
                    xscale * (s_data[threadIdx.x + 1] +
                              s_data[threadIdx.x + 11]));

        value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                    (2*zscale) * (infront6 + behind6) +
                 ((2 * pow(dx, 12) / 479001600) *
                    xscale * (s_data[threadIdx.x + 0] +
                              s_data[threadIdx.x + 12]) +
                  ((2 * pow(dx, 12) / 479001600) + (2 * pow(dt2, 12) / 479001600)) *
                    xscale * (s_data[threadIdx.x + 0] +
                              s_data[threadIdx.x + 12]));

        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("ADI Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    ADI_Optimized_time[ix] = elapsed_time;
}


__global__ void Sigma_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iz = blockDim.y * blockIdx.y + threadIdx.y;
    float xscale1, xscale2, zscale1, zscale2;

    xscale1 = sigma1 * (dt*dt) / (dx*dx);
    xscale2 = sigma2 * (dt*dt) / (dx*dx);

    zscale1 = sigma1 * (dt*dt) / (dz*dz);
    zscale2 = sigma2 * (dt*dt) / (dz*dz);

    if (ix < nx-12 && iz < nz-12) {
        int in_idx = ix + 6;
        int out_idx = 0;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz-12; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale1*(dx+dt) + xscale2*dx) + (zscale1*(dz+dt) + zscale2*dz)) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dz, 2) / 2)) *
                        (zscale1 + zscale2) * (infront1 + behind1) +
                     (((2 * pow(dx, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        xscale1 * (s_data[threadIdx.x + 5] +
                                   s_data[threadIdx.x + 7]) +
                      (2 * pow(dx, 2) / 2) *
                        xscale2 * (s_data[threadIdx.x + 5] +
                                   s_data[threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dz, 4) / 24)) *
                        (zscale1 + zscale2) * (infront2 + behind2) +
                     (((2 * pow(dx, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        xscale1 * (s_data[threadIdx.x + 4] +
                                   s_data[threadIdx.x + 8]) +
                      (2 * pow(dx, 4) / 24) *
                        xscale2 * (s_data[threadIdx.x + 4] +
                                   s_data[threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dz, 6) / 720)) *
                        (zscale1 + zscale2) * (infront3 + behind3) +
                     (((2 * pow(dx, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        xscale1 * (s_data[threadIdx.x + 3] +
                                   s_data[threadIdx.x + 9]) +
                      (2 * pow(dx, 6) / 720) *
                        xscale2 * (s_data[threadIdx.x + 3] +
                                   s_data[threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dz, 8) / 40320)) *
                        (zscale1 + zscale2) * (infront4 + behind4) +
                     (((2 * pow(dx, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        xscale1 * (s_data[threadIdx.x + 2] +
                                   s_data[threadIdx.x + 10]) +
                      (2 * pow(dx, 8) / 40320) *
                        xscale2 * (s_data[threadIdx.x + 2] +
                                   s_data[threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dz, 10) / 3628800)) *
                        (zscale1 + zscale2) * (infront5 + behind5) +
                     (((2 * pow(dx, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        xscale1 * (s_data[threadIdx.x + 1] +
                                   s_data[threadIdx.x + 11]) +
                      (2 * pow(dx, 10) / 3628800) *
                        xscale2 * (s_data[threadIdx.x + 1] +
                                   s_data[threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dz, 12) / 479001600)) *
                        (zscale1 + zscale2) * (infront6 + behind6) +
                     (((2 * pow(dx, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        xscale1 * (s_data[threadIdx.x + 0] +
                                   s_data[threadIdx.x + 12]) +
                      (2 * pow(dx, 12) / 479001600) *
                        xscale2 * (s_data[threadIdx.x + 0] +
                                   s_data[threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("Sigma Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    Sigma_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void Sigma_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    /// Precompute factor scaling to reduce overhead of floating-point
    float xscale1 = sigma1 * (dt*dt) / (dx*dx);
    float xscale2 = sigma2 * (dt*dt) / (dx*dx);

    float zscale1 = sigma1 * (dt*dt) / (dz*dz);
    float zscale2 = sigma2 * (dt*dt) / (dz*dz);

    /// Macro to 2D Grid-Stride Loop
    PlaneThreads2D_GridStrideLoop {
        int in_idx = ix + 6 + iz * nx;
        int out_idx = in_idx;
        int stride = nx;

        /// Load elemen from global memory to register
        float behind6 = d_pn[in_idx - 6 * stride];
        float behind5 = d_pn[in_idx - 5 * stride];
        float behind4 = d_pn[in_idx - 4 * stride];
        float behind3 = d_pn[in_idx - 3 * stride];
        float behind2 = d_pn[in_idx - 2 * stride];
        float behind1 = d_pn[in_idx - 1 * stride];
        float current = d_pn[in_idx];
        float infront1 = d_pn[in_idx + 1 * stride];
        float infront2 = d_pn[in_idx + 2 * stride];
        float infront3 = d_pn[in_idx + 3 * stride];
        float infront4 = d_pn[in_idx + 4 * stride];
        float infront5 = d_pn[in_idx + 5 * stride];
        float infront6 = d_pn[in_idx + 6 * stride];

        /// Shared memory
        __syncthreads();
        if (threadIdx.x < 6) {
            s_data[threadIdx.x] = d_pn[out_idx - 6];
            s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
        }
        s_data[threadIdx.x + 6] = current;
        __syncthreads();

        float value = ((xscale1*(dx+dt) + xscale2*dx) + (zscale1*(dz+dt) + zscale2*dz)) * current;

        value += ((2 * pow(dz, 2) / 2) + (2 * pow(dz, 2) / 2)) *
                    (zscale1 + zscale2) * (infront1 + behind1) +
                 (((2 * pow(dx, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                    xscale1 * (s_data[threadIdx.x + 5] +
                               s_data[threadIdx.x + 7]) +
                  (2 * pow(dx, 2) / 2) *
                    xscale2 * (s_data[threadIdx.x + 5] +
                               s_data[threadIdx.x + 7]));

        value += ((2 * pow(dz, 4) / 24) + (2 * pow(dz, 4) / 24)) *
                    (zscale1 + zscale2) * (infront2 + behind2) +
                 (((2 * pow(dx, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                    xscale1 * (s_data[threadIdx.x + 4] +
                               s_data[threadIdx.x + 8]) +
                  (2 * pow(dx, 4) / 24) *
                    xscale2 * (s_data[threadIdx.x + 4] +
                               s_data[threadIdx.x + 8]));

        value += ((2 * pow(dz, 6) / 720) + (2 * pow(dz, 6) / 720)) *
                    (zscale1 + zscale2) * (infront3 + behind3) +
                 (((2 * pow(dx, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                    xscale1 * (s_data[threadIdx.x + 3] +
                               s_data[threadIdx.x + 9]) +
                  (2 * pow(dx, 6) / 720) *
                    xscale2 * (s_data[threadIdx.x + 3] +
                               s_data[threadIdx.x + 9]));

        value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dz, 8) / 40320)) *
                    (zscale1 + zscale2) * (infront4 + behind4) +
                 (((2 * pow(dx, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                    xscale1 * (s_data[threadIdx.x + 2] +
                               s_data[threadIdx.x + 10]) +
                  (2 * pow(dx, 8) / 40320) *
                    xscale2 * (s_data[threadIdx.x + 2] +
                               s_data[threadIdx.x + 10]));

        value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dz, 10) / 3628800)) *
                    (zscale1 + zscale2) * (infront5 + behind5) +
                 (((2 * pow(dx, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                    xscale1 * (s_data[threadIdx.x + 1] +
                               s_data[threadIdx.x + 11]) +
                  (2 * pow(dx, 10) / 3628800) *
                    xscale2 * (s_data[threadIdx.x + 1] +
                               s_data[threadIdx.x + 11]));

        value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dz, 12) / 479001600)) *
                    (zscale1 + zscale2) * (infront6 + behind6) +
                 (((2 * pow(dx, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                    xscale1 * (s_data[threadIdx.x + 0] +
                               s_data[threadIdx.x + 12]) +
                  (2 * pow(dx, 12) / 479001600) *
                    xscale2 * (s_data[threadIdx.x + 0] +
                               s_data[threadIdx.x + 12]));

        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("Sigma Optimized Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    Sigma_Optimized_time[ix] = elapsed_time;
}


__global__ void LaxWendroff_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iz = blockDim.y * blockIdx.y + threadIdx.y;
    float xscale_courant, xscale_diffusion;
    float zscale_courant, zscale_diffusion;

    xscale_courant = 0.5 * dt / (dx*dx);
    xscale_diffusion = 0.5 * (dt*dt) / (dx*dx);

    zscale_courant = 0.5 * dt / (dz*dz);
    zscale_diffusion = 0.5 * (dt*dt) / (dz*dz);

    if (ix < nx-12 && iz < nz-12) {
        int in_idx = ix + 6;
        int out_idx = 0;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz-6; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale_diffusion*dx + zscale_diffusion*dz) +
                           (xscale_courant*dx + zscale_courant*dz)) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dz, 1) / 1)) *
                        (zscale_courant+zscale_diffusion) * (infront1 + behind1) +
                     ((2 * pow(dx, 1) / 1) *
                        xscale_courant * (s_data[threadIdx.x + 5] +
                                          s_data[threadIdx.x + 7]) +
                      (2 * pow(dx, 2) / 2) *
                        xscale_diffusion * (s_data[threadIdx.x + 5] +
                                            s_data[threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dz, 3) / 6)) *
                        (zscale_courant+zscale_diffusion) * (infront2 + behind2) +
                     ((2 * pow(dx, 3) / 6) *
                        xscale_courant * (s_data[threadIdx.x + 4] +
                                          s_data[threadIdx.x + 8]) +
                      (2 * pow(dx, 4) / 24) *
                        xscale_diffusion * (s_data[threadIdx.x + 4] +
                                            s_data[threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dz, 5) / 120)) *
                        (zscale_courant+zscale_diffusion) * (infront3 + behind3) +
                     ((2 * pow(dx, 5) / 120) *
                        xscale_courant * (s_data[threadIdx.x + 3] +
                                          s_data[threadIdx.x + 9]) +
                      (2 * pow(dx, 6) / 720) *
                        xscale_diffusion * (s_data[threadIdx.x + 3] +
                                            s_data[threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dz, 7) / 5040)) *
                        (zscale_courant+zscale_diffusion) * (infront4 + behind4) +
                     ((2 * pow(dx, 7) / 5040) *
                        xscale_courant * (s_data[threadIdx.x + 2] +
                                          s_data[threadIdx.x + 10]) +
                      (2 * pow(dx, 8) / 40320) *
                        xscale_diffusion * (s_data[threadIdx.x + 2] +
                                            s_data[threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dz, 9) / 362880)) *
                        (zscale_courant+zscale_diffusion) * (infront5 + behind5) +
                     ((2 * pow(dx, 9) / 5040) *
                        xscale_courant * (s_data[threadIdx.x + 1] +
                                          s_data[threadIdx.x + 11]) +
                      (2 * pow(dx, 10) / 3628800) *
                        xscale_diffusion * (s_data[threadIdx.x + 1] +
                                            s_data[threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dz, 11) / 39916800)) *
                        (zscale_courant+zscale_diffusion) * (infront6 + behind6) +
                     ((2 * pow(dx, 11) / 39916800) *
                        xscale_courant * (s_data[threadIdx.x + 0] +
                                          s_data[threadIdx.x + 12]) +
                      (2 * pow(dx, 12) / 479001600) *
                        xscale_diffusion * (s_data[threadIdx.x + 0] +
                                            s_data[threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("LaxWendroff Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    LaxWendroff_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void LaxWendroff_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    /// Precompute factor scaling to reduce overhead of floating-point
    float xscale_courant = 0.5 * dt / (dx*dx);
    float xscale_diffusion = 0.5 * (dt*dt) / (dx*dx);

    float zscale_courant = 0.5 * dt / (dz*dz);
    float zscale_diffusion = 0.5 * (dt*dt) / (dz*dz);

    /// Macro to 2D Grid-Stride Loop
    PlaneThreads2D_GridStrideLoop {
        int in_idx = ix + 6 + iz * nx;
        int out_idx = in_idx;
        int stride = nx;

        /// Load elemen from global memory to register
        float behind6 = d_pn[in_idx - 6 * stride];
        float behind5 = d_pn[in_idx - 5 * stride];
        float behind4 = d_pn[in_idx - 4 * stride];
        float behind3 = d_pn[in_idx - 3 * stride];
        float behind2 = d_pn[in_idx - 2 * stride];
        float behind1 = d_pn[in_idx - 1 * stride];
        float current = d_pn[in_idx];
        float infront1 = d_pn[in_idx + 1 * stride];
        float infront2 = d_pn[in_idx + 2 * stride];
        float infront3 = d_pn[in_idx + 3 * stride];
        float infront4 = d_pn[in_idx + 4 * stride];
        float infront5 = d_pn[in_idx + 5 * stride];
        float infront6 = d_pn[in_idx + 6 * stride];

        /// Shared memory
        __syncthreads();
        if (threadIdx.x < 6) {
            s_data[threadIdx.x] = d_pn[out_idx - 6];
            s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
        }
        s_data[threadIdx.x + 6] = current;
        __syncthreads();

        float value = ((xscale_diffusion*dx + zscale_diffusion*dz) +
                       (xscale_courant*dx + zscale_courant*dz)) * current;

        value += ((2 * pow(dz, 2) / 2) + (2 * pow(dz, 1) / 1)) *
                    (zscale_courant+zscale_diffusion) * (infront1 + behind1) +
                 ((2 * pow(dx, 1) / 1) *
                    xscale_courant * (s_data[threadIdx.x + 5] +
                                      s_data[threadIdx.x + 7]) +
                  (2 * pow(dx, 2) / 2) *
                    xscale_diffusion * (s_data[threadIdx.x + 5] +
                                        s_data[threadIdx.x + 7]));

        value += ((2 * pow(dz, 4) / 24) + (2 * pow(dz, 3) / 6)) *
                    (zscale_courant+zscale_diffusion) * (infront2 + behind2) +
                 ((2 * pow(dx, 3) / 6) *
                    xscale_courant * (s_data[threadIdx.x + 4] +
                                      s_data[threadIdx.x + 8]) +
                  (2 * pow(dx, 4) / 24) *
                    xscale_diffusion * (s_data[threadIdx.x + 4] +
                                        s_data[threadIdx.x + 8]));

        value += ((2 * pow(dz, 6) / 720) + (2 * pow(dz, 5) / 120)) *
                    (zscale_courant+zscale_diffusion) * (infront3 + behind3) +
                 ((2 * pow(dx, 5) / 120) *
                    xscale_courant * (s_data[threadIdx.x + 3] +
                                      s_data[threadIdx.x + 9]) +
                  (2 * pow(dx, 6) / 720) *
                    xscale_diffusion * (s_data[threadIdx.x + 3] +
                                        s_data[threadIdx.x + 9]));

        value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dz, 7) / 5040)) *
                    (zscale_courant+zscale_diffusion) * (infront4 + behind4) +
                 ((2 * pow(dx, 7) / 5040) *
                    xscale_courant * (s_data[threadIdx.x + 2] +
                                      s_data[threadIdx.x + 10]) +
                  (2 * pow(dx, 8) / 40320) *
                    xscale_diffusion * (s_data[threadIdx.x + 2] +
                                        s_data[threadIdx.x + 10]));

        value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dz, 9) / 362880)) *
                    (zscale_courant+zscale_diffusion) * (infront5 + behind5) +
                 ((2 * pow(dx, 9) / 5040) *
                    xscale_courant * (s_data[threadIdx.x + 1] +
                                      s_data[threadIdx.x + 11]) +
                  (2 * pow(dx, 10) / 3628800) *
                    xscale_diffusion * (s_data[threadIdx.x + 1] +
                                        s_data[threadIdx.x + 11]));

        value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dz, 11) / 39916800)) *
                    (zscale_courant+zscale_diffusion) * (infront6 + behind6) +
                 ((2 * pow(dx, 11) / 39916800) *
                    xscale_courant * (s_data[threadIdx.x + 0] +
                                      s_data[threadIdx.x + 12]) +
                  (2 * pow(dx, 12) / 479001600) *
                    xscale_diffusion * (s_data[threadIdx.x + 0] +
                                        s_data[threadIdx.x + 12]));

        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("LaxWendroff Optimized Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    LaxWendroff_Optimized_time[ix] = elapsed_time;
}


__global__ void FractionalStep_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iz = blockDim.y * blockIdx.y + threadIdx.y;
    float xscale, zscale;
    float dt2 = 0.5*dt;

    xscale = 0.5 * (dt*dt*0.5) / (dx*dx);
    zscale = 0.5 * (dt*dt*0.5) / (dz*dz);

    if (ix < nx-12 && iz < nz-12) {
        int in_idx = ix + 6;
        int out_idx = 0;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz-6; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale*(dx+dt2) + zscale*(dz+dt)) + (xscale*dx + zscale*dz)) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        (4*zscale) * (infront1 + behind1) +
                     (((2 * pow(dx, 2) / 2) + (2 * pow(dt2, 2) / 2)) *
                        (2*xscale) * (s_data[threadIdx.x + 5] +
                                      s_data[threadIdx.x + 7]) +
                      (2 * pow(dx, 2) / 2) *
                        (2*xscale) * (s_data[threadIdx.x + 5] +
                                      s_data[threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        (4*zscale) * (infront2 + behind2) +
                     (((2 * pow(dx, 4) / 24) + (2 * pow(dt2, 4) / 24)) *
                        (2*xscale) * (s_data[threadIdx.x + 4] +
                                      s_data[threadIdx.x + 8]) +
                      (2 * pow(dx, 4) / 24) *
                        (2*xscale) * (s_data[threadIdx.x + 4] +
                                      s_data[threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        (4*zscale) * (infront3 + behind3) +
                     (((2 * pow(dx, 6) / 720) + (2 * pow(dt2, 6) / 720)) *
                        (2*xscale) * (s_data[threadIdx.x + 3] +
                                      s_data[threadIdx.x + 9]) +
                      (2 * pow(dx, 6) / 720) *
                        (2*xscale) * (s_data[threadIdx.x + 3] +
                                      s_data[threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        (4*zscale) * (infront4 + behind4) +
                     (((2 * pow(dx, 8) / 40320) + (2 * pow(dt2, 8) / 40320)) *
                        (2*xscale) * (s_data[threadIdx.x + 2] +
                                      s_data[threadIdx.x + 10]) +
                      (2 * pow(dx, 8) / 40320) *
                        (2*xscale) * (s_data[threadIdx.x + 2] +
                                      s_data[threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        (4*zscale) * (infront5 + behind5) +
                     (((2 * pow(dx, 10) / 3628800) + (2 * pow(dt2, 10) / 3628800)) *
                        (2*xscale) * (s_data[threadIdx.x + 1] +
                                      s_data[threadIdx.x + 11]) +
                      (2 * pow(dx, 10) / 3628800) *
                        (2*xscale) * (s_data[threadIdx.x + 1] +
                                      s_data[threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        (4*zscale) * (infront6 + behind6) +
                     (((2 * pow(dx, 12) / 479001600) + (2 * pow(dt2, 12) / 479001600)) *
                        (2*xscale) * (s_data[threadIdx.x + 0] +
                                      s_data[threadIdx.x + 12]) +
                      (2 * pow(dx, 12) / 479001600) *
                        (2*xscale) * (s_data[threadIdx.x + 0] +
                                      s_data[threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("FractionalStep Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    FractionalStep_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void FractionalStep_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    float dt2 = 0.5*dt;

    /// Precompute factor scaling to reduce overhead of floating-point
    float xscale = 0.5 * (dt*dt*0.5) / (dx*dx);
    float zscale = 0.5 * (dt*dt*0.5) / (dz*dz);

    /// Macro to 2D Grid-Stride Loop
    PlaneThreads2D_GridStrideLoop {
        int in_idx = ix + 6 + iz * nx;
        int out_idx = in_idx;
        int stride = nx;

        /// Load elemen from global memory to register
        float behind6 = d_pn[in_idx - 6 * stride];
        float behind5 = d_pn[in_idx - 5 * stride];
        float behind4 = d_pn[in_idx - 4 * stride];
        float behind3 = d_pn[in_idx - 3 * stride];
        float behind2 = d_pn[in_idx - 2 * stride];
        float behind1 = d_pn[in_idx - 1 * stride];
        float current = d_pn[in_idx];
        float infront1 = d_pn[in_idx + 1 * stride];
        float infront2 = d_pn[in_idx + 2 * stride];
        float infront3 = d_pn[in_idx + 3 * stride];
        float infront4 = d_pn[in_idx + 4 * stride];
        float infront5 = d_pn[in_idx + 5 * stride];
        float infront6 = d_pn[in_idx + 6 * stride];

        /// Shared memory
        __syncthreads();
        if (threadIdx.x < 6) {
            s_data[threadIdx.x] = d_pn[out_idx - 6];
            s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
        }
        s_data[threadIdx.x + 6] = current;
        __syncthreads();

        float value = ((xscale*(dx+dt2) + zscale*(dz+dt)) + (xscale*dx + zscale*dz)) * current;

        value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                    (4*zscale) * (infront1 + behind1) +
                 (((2 * pow(dx, 2) / 2) + (2 * pow(dt2, 2) / 2)) *
                    (2*xscale) * (s_data[threadIdx.x + 5] +
                                  s_data[threadIdx.x + 7]) +
                  (2 * pow(dx, 2) / 2) *
                    (2*xscale) * (s_data[threadIdx.x + 5] +
                                  s_data[threadIdx.x + 7]));

        value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                    (4*zscale) * (infront2 + behind2) +
                 (((2 * pow(dx, 4) / 24) + (2 * pow(dt2, 4) / 24)) *
                    (2*xscale) * (s_data[threadIdx.x + 4] +
                                  s_data[threadIdx.x + 8]) +
                  (2 * pow(dx, 4) / 24) *
                    (2*xscale) * (s_data[threadIdx.x + 4] +
                                  s_data[threadIdx.x + 8]));

        value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                    (4*zscale) * (infront3 + behind3) +
                 (((2 * pow(dx, 6) / 720) + (2 * pow(dt2, 6) / 720)) *
                    (2*xscale) * (s_data[threadIdx.x + 3] +
                                  s_data[threadIdx.x + 9]) +
                  (2 * pow(dx, 6) / 720) *
                    (2*xscale) * (s_data[threadIdx.x + 3] +
                                  s_data[threadIdx.x + 9]));

        value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                    (4*zscale) * (infront4 + behind4) +
                 (((2 * pow(dx, 8) / 40320) + (2 * pow(dt2, 8) / 40320)) *
                    (2*xscale) * (s_data[threadIdx.x + 2] +
                                  s_data[threadIdx.x + 10]) +
                  (2 * pow(dx, 8) / 40320) *
                    (2*xscale) * (s_data[threadIdx.x + 2] +
                                  s_data[threadIdx.x + 10]));

        value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                    (4*zscale) * (infront5 + behind5) +
                 (((2 * pow(dx, 10) / 3628800) + (2 * pow(dt2, 10) / 3628800)) *
                    (2*xscale) * (s_data[threadIdx.x + 1] +
                                  s_data[threadIdx.x + 11]) +
                  (2 * pow(dx, 10) / 3628800) *
                    (2*xscale) * (s_data[threadIdx.x + 1] +
                                  s_data[threadIdx.x + 11]));

        value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                    (4*zscale) * (infront6 + behind6) +
                 (((2 * pow(dx, 12) / 479001600) + (2 * pow(dt2, 12) / 479001600)) *
                    (2*xscale) * (s_data[threadIdx.x + 0] +
                                  s_data[threadIdx.x + 12]) +
                  (2 * pow(dx, 12) / 479001600) *
                    (2*xscale) * (s_data[threadIdx.x + 0] +
                                  s_data[threadIdx.x + 12]));

        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("FractionalStep Optimized Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    FractionalStep_Optimized_time[ix] = elapsed_time;
}


__global__ void MacCormack_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iz = blockDim.y * blockIdx.y + threadIdx.y;
    float xscale_predictor, xscale_corrector;
    float zscale_predictor, zscale_corrector;

    xscale_predictor = 0.5 * dt / (dx*dx);
    xscale_corrector = (dt*dt) / (dx*dx);

    zscale_predictor = 0.5 * dt / (dz*dz);
    zscale_corrector = (dt*dt) / (dz*dz);

    if (ix < nx-12 && iz < nz-12) {
        int in_idx = ix + 6;
        int out_idx = 0;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz-6; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale_corrector*(dx+dt) + zscale_corrector*(dz+dt)) +
                           (xscale_predictor*dx + zscale_predictor*dz)) * current;

            value += (((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        zscale_corrector * (infront1 + behind1) +
                      (2 * pow(dz, 1) / 1) *
                        zscale_predictor * (infront1 + behind1)) +
                     (((2 * pow(dx, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        xscale_corrector * (s_data[threadIdx.x + 5] +
                                            s_data[threadIdx.x + 7]) +
                      (2 * pow(dx, 1) / 1) *
                        xscale_predictor * (s_data[threadIdx.x + 5] +
                                            s_data[threadIdx.x + 7]));

            value += (((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        zscale_corrector * (infront2 + behind2) +
                      (2 * pow(dz, 3) / 6) *
                        zscale_predictor * (infront2 + behind2)) +
                     (((2 * pow(dx, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        xscale_corrector * (s_data[threadIdx.x + 4] +
                                            s_data[threadIdx.x + 8]) +
                      (2 * pow(dx, 3) / 6) *
                        xscale_predictor * (s_data[threadIdx.x + 4] +
                                            s_data[threadIdx.x + 8]));

            value += (((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        zscale_corrector * (infront3 + behind3) +
                      (2 * pow(dz, 5) / 120) *
                        zscale_predictor * (infront3 + behind3)) +
                     (((2 * pow(dx, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        xscale_corrector * (s_data[threadIdx.x + 3] +
                                            s_data[threadIdx.x + 9]) +
                      (2 * pow(dx, 5) / 120) *
                        xscale_predictor * (s_data[threadIdx.x + 3] +
                                            s_data[threadIdx.x + 9]));

            value += (((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        zscale_corrector * (infront4 + behind4) +
                      (2 * pow(dz, 7) / 5040) *
                        zscale_predictor * (infront4 + behind4)) +
                     (((2 * pow(dx, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        xscale_corrector * (s_data[threadIdx.x + 2] +
                                            s_data[threadIdx.x + 10]) +
                      (2 * pow(dx, 7) / 5040) *
                        xscale_predictor * (s_data[threadIdx.x + 2] +
                                            s_data[threadIdx.x + 10]));

            value += (((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        zscale_corrector * (infront5 + behind5) +
                      (2 * pow(dz, 9) / 362880) *
                        zscale_predictor * (infront5 + behind5)) +
                     (((2 * pow(dx, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        xscale_corrector * (s_data[threadIdx.x + 1] +
                                            s_data[threadIdx.x + 11]) +
                      (2 * pow(dx, 9) / 362880) *
                        xscale_predictor * (s_data[threadIdx.x + 1] +
                                            s_data[threadIdx.x + 11]));

            value += (((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        zscale_corrector * (infront6 + behind6) +
                      (2 * pow(dz, 11) / 39916800) *
                        zscale_predictor * (infront6 + behind6)) +
                     (((2 * pow(dx, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        xscale_corrector * (s_data[threadIdx.x + 0] +
                                            s_data[threadIdx.x + 12]) +
                      (2 * pow(dx, 11) / 39916800) *
                        xscale_predictor * (s_data[threadIdx.x + 0] +
                                            s_data[threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("MacCormack Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    MacCormack_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void MacCormack_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    /// Precompute factor scaling to reduce overhead of floating-point
    float xscale_predictor = 0.5 * dt / (dx*dx);
    float xscale_corrector = (dt*dt) / (dx*dx);

    float zscale_predictor = 0.5 * dt / (dz*dz);
    float zscale_corrector = (dt*dt) / (dz*dz);

    /// Macro to 2D Grid-Stride Loop
    PlaneThreads2D_GridStrideLoop {
        int in_idx = ix + 6 + iz * nx;
        int out_idx = in_idx;
        int stride = nx;

        /// Load elemen from global memory to register
        float behind6 = d_pn[in_idx - 6 * stride];
        float behind5 = d_pn[in_idx - 5 * stride];
        float behind4 = d_pn[in_idx - 4 * stride];
        float behind3 = d_pn[in_idx - 3 * stride];
        float behind2 = d_pn[in_idx - 2 * stride];
        float behind1 = d_pn[in_idx - 1 * stride];
        float current = d_pn[in_idx];
        float infront1 = d_pn[in_idx + 1 * stride];
        float infront2 = d_pn[in_idx + 2 * stride];
        float infront3 = d_pn[in_idx + 3 * stride];
        float infront4 = d_pn[in_idx + 4 * stride];
        float infront5 = d_pn[in_idx + 5 * stride];
        float infront6 = d_pn[in_idx + 6 * stride];

        /// Shared memory
        __syncthreads();
        if (threadIdx.x < 6) {
            s_data[threadIdx.x] = d_pn[out_idx - 6];
            s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
        }
        s_data[threadIdx.x + 6] = current;
        __syncthreads();

        float value = ((xscale_corrector*(dx+dt) + zscale_corrector*(dz+dt)) +
                       (xscale_predictor*dx + zscale_predictor*dz)) * current;

        value += (((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                    zscale_corrector * (infront1 + behind1) +
                  (2 * pow(dz, 1) / 1) *
                    zscale_predictor * (infront1 + behind1)) +
                 (((2 * pow(dx, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                    xscale_corrector * (s_data[threadIdx.x + 5] +
                                        s_data[threadIdx.x + 7]) +
                  (2 * pow(dx, 1) / 1) *
                    xscale_predictor * (s_data[threadIdx.x + 5] +
                                        s_data[threadIdx.x + 7]));

        value += (((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                    zscale_corrector * (infront2 + behind2) +
                  (2 * pow(dz, 3) / 6) *
                    zscale_predictor * (infront2 + behind2)) +
                 (((2 * pow(dx, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                    xscale_corrector * (s_data[threadIdx.x + 4] +
                                        s_data[threadIdx.x + 8]) +
                  (2 * pow(dx, 3) / 6) *
                    xscale_predictor * (s_data[threadIdx.x + 4] +
                                        s_data[threadIdx.x + 8]));

        value += (((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                    zscale_corrector * (infront3 + behind3) +
                  (2 * pow(dz, 5) / 120) *
                    zscale_predictor * (infront3 + behind3)) +
                 (((2 * pow(dx, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                    xscale_corrector * (s_data[threadIdx.x + 3] +
                                        s_data[threadIdx.x + 9]) +
                  (2 * pow(dx, 5) / 120) *
                    xscale_predictor * (s_data[threadIdx.x + 3] +
                                        s_data[threadIdx.x + 9]));

        value += (((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                    zscale_corrector * (infront4 + behind4) +
                  (2 * pow(dz, 7) / 5040) *
                    zscale_predictor * (infront4 + behind4)) +
                 (((2 * pow(dx, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                    xscale_corrector * (s_data[threadIdx.x + 2] +
                                        s_data[threadIdx.x + 10]) +
                  (2 * pow(dx, 7) / 5040) *
                    xscale_predictor * (s_data[threadIdx.x + 2] +
                                        s_data[threadIdx.x + 10]));

        value += (((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                    zscale_corrector * (infront5 + behind5) +
                  (2 * pow(dz, 9) / 362880) *
                    zscale_predictor * (infront5 + behind5)) +
                 (((2 * pow(dx, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                    xscale_corrector * (s_data[threadIdx.x + 1] +
                                        s_data[threadIdx.x + 11]) +
                  (2 * pow(dx, 9) / 362880) *
                    xscale_predictor * (s_data[threadIdx.x + 1] +
                                        s_data[threadIdx.x + 11]));

        value += (((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                    zscale_corrector * (infront6 + behind6) +
                  (2 * pow(dz, 11) / 39916800) *
                    zscale_predictor * (infront6 + behind6)) +
                 (((2 * pow(dx, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                    xscale_corrector * (s_data[threadIdx.x + 0] +
                                        s_data[threadIdx.x + 12]) +
                  (2 * pow(dx, 11) / 39916800) *
                    xscale_predictor * (s_data[threadIdx.x + 0] +
                                        s_data[threadIdx.x + 12]));

        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("MacCormack Optimized Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    MacCormack_Optimized_time[ix] = elapsed_time;
}


__global__ void TVD_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iz = blockDim.y * blockIdx.y + threadIdx.y;
    float xscale_first, xscale_second, zscale_first, zscale_second;

    xscale_first = 0.5 * dt / dx;
    xscale_second = (dt*dt) / (dx*dx);

    zscale_first = 0.5 * dt / dz;
    zscale_second = (dt*dt) / (dz*dz);

    if (ix < nx-12 && iz < nz-12) {
        int in_idx = ix + 6;
        int out_idx = 0;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        /// Define index thread
        int tx = threadIdx.x + 6;

        /// Symmetric Flux Limiters from Waterson & Deconinck, 2007.
        /// Paper: "Design principles for bounded higher-order convection schemes   a unified approach"
        /// int rx = (tx-1)/(tx+1);
        /// int flx = (1.5*(pow(rx,2)+rx)) / (pow(rx,2)+rx+1);
        int flx1 = (1.5*(pow((tx-1)/(tx+1),2)+((tx-1)/(tx+1)))) / (pow((tx-1)/(tx+1),2)+((tx-1)/(tx+1))+1);
        int flx2 = (1.5*(pow((tx-2)/(tx+2),2)+((tx-2)/(tx+2)))) / (pow((tx-2)/(tx+2),2)+((tx-2)/(tx+2))+1);
        int flx3 = (1.5*(pow((tx-3)/(tx+4),2)+((tx-3)/(tx+3)))) / (pow((tx-3)/(tx+3),2)+((tx-3)/(tx+3))+1);
        int flx4 = (1.5*(pow((tx-4)/(tx+4),2)+((tx-4)/(tx+4)))) / (pow((tx-4)/(tx+4),2)+((tx-4)/(tx+4))+1);
        int flx5 = (1.5*(pow((tx-5)/(tx+5),2)+((tx-5)/(tx+5)))) / (pow((tx-5)/(tx+5),2)+((tx-5)/(tx+5))+1);
        int flx6 = (1.5*(pow((tx-6)/(tx+6),2)+((tx-6)/(tx+6)))) / (pow((tx-6)/(tx+6),2)+((tx-6)/(tx+6))+1);

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx  += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz-6; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[tx] = current;
            __syncthreads();

            float value = ((xscale_second*dx + zscale_second*dz) +
                           (xscale_first*dx + zscale_first*dz)) * current;

            value += ((2 * pow(dz, 1) / 1) *
                        zscale_first * (infront1 + behind1) +
                      (2 * pow(dz, 2) / 2) *
                        zscale_second * (infront1 + behind1)) +
                     ((2 * pow(dx, 1) / 1) *
                        xscale_first * (s_data[tx - 1 - flx1] +
                                        s_data[tx + 1 + flx1]) +
                      (2 * pow(dx, 2) / 2) *
                        xscale_second * (s_data[tx - 1] +
                                         s_data[tx + 1]));

            value += ((2 * pow(dz, 3) / 6) *
                        zscale_first * (infront2 + behind2) +
                      (2 * pow(dz, 4) / 24) *
                        zscale_second * (infront2 + behind2)) +
                     ((2 * pow(dx, 3) / 6) *
                        xscale_first * (s_data[tx - 2 - flx2] +
                                        s_data[tx + 2 + flx2]) +
                      (2 * pow(dx, 4) / 24) *
                        xscale_second * (s_data[tx - 2] +
                                         s_data[tx + 2]));

            value += ((2 * pow(dz, 5) / 120) *
                        zscale_first * (infront3 + behind3) +
                      (2 * pow(dz, 6) / 720) *
                        zscale_second * (infront3 + behind3)) +
                     ((2 * pow(dx, 5) / 120) *
                        xscale_first * (s_data[tx - 3 - flx3] +
                                        s_data[tx + 3 + flx3]) +
                      (2 * pow(dx, 6) / 720) *
                        xscale_second * (s_data[tx - 3] +
                                         s_data[tx + 3]));

            value += ((2 * pow(dz, 7) / 5040) *
                        zscale_first * (infront4 + behind4) +
                      (2 * pow(dz, 8) / 40320) *
                        zscale_second * (infront4 + behind4)) +
                     ((2 * pow(dx, 7) / 5040) *
                        xscale_first * (s_data[tx - 4 - flx4] +
                                        s_data[tx + 4 + flx4]) +
                      (2 * pow(dx, 8) / 40320) *
                        xscale_second * (s_data[tx - 4] +
                                         s_data[tx + 4]));

            value += ((2 * pow(dz, 9) / 362880) *
                        zscale_first * (infront5 + behind5) +
                      (2 * pow(dz, 10) / 3628800) *
                        zscale_second * (infront5 + behind5)) +
                     ((2 * pow(dx, 9) / 362880) *
                        xscale_first * (s_data[tx - 5 - flx5] +
                                        s_data[tx + 5 + flx5]) +
                      (2 * pow(dx, 10) / 3628800) *
                        xscale_second * (s_data[tx - 5] +
                                         s_data[tx + 5]));

            value += ((2 * pow(dz, 11) / 39916800) *
                        zscale_first * (infront6 + behind6) +
                      (2 * pow(dz, 12) / 479001600) *
                        zscale_second * (infront6 + behind6)) +
                     ((2 * pow(dx, 11) / 39916800) *
                        xscale_first * (s_data[tx - 6 - flx6] +
                                        s_data[tx + 6 + flx6]) +
                      (2 * pow(dx, 12) / 479001600) *
                        xscale_second * (s_data[tx - 6] +
                                         s_data[tx + 6]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("TVD Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    TVD_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void TVD_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    /// Precompute factor scaling to reduce overhead of floating-point
    float xscale_first = 0.5 * dt / dx;
    float xscale_second = (dt*dt) / (dx*dx);

    float zscale_first = 0.5 * dt / dz;
    float zscale_second = (dt*dt) / (dz*dz);

    /// Macro to 2D Grid-Stride Loop
    PlaneThreads2D_GridStrideLoop {
        int in_idx = ix + 6 + iz * nx;
        int out_idx = in_idx;
        int stride = nx;

        /// Define index thread
        int tx = threadIdx.x + 6;

        /// Symmetric Flux Limiters from Waterson & Deconinck, 2007.
        /// Paper: "Design principles for bounded higher-order convection schemes: a unified approach"
        /// int rx = (tx-1)/(tx+1);
        /// int flx = (1.5*(pow(rx,2)+rx)) / (pow(rx,2)+rx+1);
        int flx1 = (1.5*(pow((tx-1)/(tx+1),2)+((tx-1)/(tx+1)))) / (pow((tx-1)/(tx+1),2)+((tx-1)/(tx+1))+1);
        int flx2 = (1.5*(pow((tx-2)/(tx+2),2)+((tx-2)/(tx+2)))) / (pow((tx-2)/(tx+2),2)+((tx-2)/(tx+2))+1);
        int flx3 = (1.5*(pow((tx-3)/(tx+4),2)+((tx-3)/(tx+3)))) / (pow((tx-3)/(tx+3),2)+((tx-3)/(tx+3))+1);
        int flx4 = (1.5*(pow((tx-4)/(tx+4),2)+((tx-4)/(tx+4)))) / (pow((tx-4)/(tx+4),2)+((tx-4)/(tx+4))+1);
        int flx5 = (1.5*(pow((tx-5)/(tx+5),2)+((tx-5)/(tx+5)))) / (pow((tx-5)/(tx+5),2)+((tx-5)/(tx+5))+1);
        int flx6 = (1.5*(pow((tx-6)/(tx+6),2)+((tx-6)/(tx+6)))) / (pow((tx-6)/(tx+6),2)+((tx-6)/(tx+6))+1);

        /// Load elemen from global memory to register
        float behind6 = d_pn[in_idx - 6 * stride];
        float behind5 = d_pn[in_idx - 5 * stride];
        float behind4 = d_pn[in_idx - 4 * stride];
        float behind3 = d_pn[in_idx - 3 * stride];
        float behind2 = d_pn[in_idx - 2 * stride];
        float behind1 = d_pn[in_idx - 1 * stride];
        float current = d_pn[in_idx];
        float infront1 = d_pn[in_idx + 1 * stride];
        float infront2 = d_pn[in_idx + 2 * stride];
        float infront3 = d_pn[in_idx + 3 * stride];
        float infront4 = d_pn[in_idx + 4 * stride];
        float infront5 = d_pn[in_idx + 5 * stride];
        float infront6 = d_pn[in_idx + 6 * stride];

        /// Shared memory
        __syncthreads();
        if (threadIdx.x < 6) {
            s_data[threadIdx.x] = d_pn[out_idx - 6];
            s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
        }
        s_data[threadIdx.x + 6] = current;
        __syncthreads();

        float value = ((xscale_second*dx + zscale_second*dz) +
                       (xscale_first*dx + zscale_first*dz)) * current;

        value += ((2 * pow(dz, 1) / 1) *
                    zscale_first * (infront1 + behind1) +
                  (2 * pow(dz, 2) / 2) *
                    zscale_second * (infront1 + behind1)) +
                 ((2 * pow(dx, 1) / 1) *
                    xscale_first * (s_data[tx - 1 - flx1] +
                                    s_data[tx + 1 + flx1]) +
                  (2 * pow(dx, 2) / 2) *
                    xscale_second * (s_data[tx - 1] +
                                     s_data[tx + 1]));

        value += ((2 * pow(dz, 3) / 6) *
                    zscale_first * (infront2 + behind2) +
                  (2 * pow(dz, 4) / 24) *
                    zscale_second * (infront2 + behind2)) +
                 ((2 * pow(dx, 3) / 6) *
                    xscale_first * (s_data[tx - 2 - flx2] +
                                    s_data[tx + 2 + flx2]) +
                  (2 * pow(dx, 4) / 24) *
                    xscale_second * (s_data[tx - 2] +
                                     s_data[tx + 2]));

        value += ((2 * pow(dz, 5) / 120) *
                    zscale_first * (infront3 + behind3) +
                  (2 * pow(dz, 6) / 720) *
                    zscale_second * (infront3 + behind3)) +
                 ((2 * pow(dx, 5) / 120) *
                    xscale_first * (s_data[tx - 3 - flx3] +
                                    s_data[tx + 3 + flx3]) +
                  (2 * pow(dx, 6) / 720) *
                    xscale_second * (s_data[tx - 3] +
                                     s_data[tx + 3]));

        value += ((2 * pow(dz, 7) / 5040) *
                    zscale_first * (infront4 + behind4) +
                  (2 * pow(dz, 8) / 40320) *
                    zscale_second * (infront4 + behind4)) +
                 ((2 * pow(dx, 7) / 5040) *
                    xscale_first * (s_data[tx - 4 - flx4] +
                                    s_data[tx + 4 + flx4]) +
                  (2 * pow(dx, 8) / 40320) *
                    xscale_second * (s_data[tx - 4] +
                                     s_data[tx + 4]));

        value += ((2 * pow(dz, 9) / 362880) *
                    zscale_first * (infront5 + behind5) +
                  (2 * pow(dz, 10) / 3628800) *
                    zscale_second * (infront5 + behind5)) +
                 ((2 * pow(dx, 9) / 362880) *
                    xscale_first * (s_data[tx - 5 - flx5] +
                                    s_data[tx + 5 + flx5]) +
                  (2 * pow(dx, 10) / 3628800) *
                    xscale_second * (s_data[tx - 5] +
                                     s_data[tx + 5]));

        value += ((2 * pow(dz, 11) / 39916800) *
                    zscale_first * (infront6 + behind6) +
                  (2 * pow(dz, 12) / 479001600) *
                    zscale_second * (infront6 + behind6)) +
                 ((2 * pow(dx, 11) / 39916800) *
                    xscale_first * (s_data[tx - 6 - flx6] +
                                    s_data[tx + 6 + flx6]) +
                  (2 * pow(dx, 12) / 479001600) *
                    xscale_second * (s_data[tx - 6] +
                                     s_data[tx + 6]));

        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("TVD Optimized Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    TVD_Optimized_time[ix] = elapsed_time;
}


__global__ void PSOR_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iz = blockDim.y * blockIdx.y + threadIdx.y;
    float xscale, zscale;
    float first_scale, second_scale, sor, a, wopt;

    xscale = (dt*dt) / (dx*dx);
    zscale = (dt*dt) / (dz*dz);

    /// Based on Hoffmann(2000) approximation
    sor = xscale / zscale;
    a = powf((cos(M_PI/(IM-1)) + (sor*sor)*cos(M_PI/(JM-1))) / (1+sor*sor), 2);
    wopt = (2-2*sqrt(1-a))/a;

    first_scale = wopt / (2*(1+sor*sor));
    second_scale = wopt*(sor*sor) / (2*(1+sor*sor));

    if (ix < nx-12 && iz < nz-12) {
        int in_idx = ix + 6;
        int out_idx = 0;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        float coefficient_1 = dx + dz;
        float coefficient_2 = 2*pow(dx,2)/2 + 2*pow(dz,2)/2;
        float coefficient_3 = 2*pow(dx,4)/24 + 2*pow(dz,4)/24;
        float coefficient_4 = 2*pow(dx,6)/720 + 2*pow(dz,6)/720;
        float coefficient_5 = 2*pow(dx,8)/40320 + 2*pow(dz,8)/40320;
        float coefficient_6 = 2*pow(dx,10)/3628800 + 2*pow(dz,10)/3628800;
        float coefficient_7 = 2*pow(dx,12)/479001600 + 2*pow(dz,12)/479001600;

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz-6; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((first_scale+second_scale) * wopt) * (coefficient_1) * current;

            value += coefficient_2 * (first_scale * (infront1 + behind1) +
                                      second_scale * (s_data[threadIdx.x + 5] +
                                                      s_data[threadIdx.x + 7]));

            value += coefficient_3 * (first_scale * (infront2 + behind2) +
                                      second_scale * (s_data[threadIdx.x + 4] +
                                                      s_data[threadIdx.x + 8]));

            value += coefficient_4 * (first_scale * (infront3 + behind3) +
                                      second_scale * (s_data[threadIdx.x + 3] +
                                                      s_data[threadIdx.x + 9]));

            value += coefficient_5 * (first_scale * (infront4 + behind4) +
                                      second_scale * (s_data[threadIdx.x + 2] +
                                                      s_data[threadIdx.x + 10]));

            value += coefficient_6 * (first_scale * (infront5 + behind5) +
                                      second_scale * (s_data[threadIdx.x + 1] +
                                                      s_data[threadIdx.x + 11]));

            value += coefficient_7 * (first_scale * (infront6 + behind6) +
                                      second_scale * (s_data[threadIdx.x + 0] +
                                                      s_data[threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("PSOR Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    PSOR_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void PSOR_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    /// Precompute factor scaling to reduce overhead of floating-point
    float xscale = (dt*dt) / (dx*dx);
    float zscale = (dt*dt) / (dz*dz);

    /// Based on Hoffmann(2000) approximation
    float sor = xscale / zscale;
    float a = powf((cos(M_PI/(IM-1)) + (sor*sor)*cos(M_PI/(JM-1))) / (1+sor*sor), 2);
    float wopt = (2-2*sqrt(1-a))/a;

    float first_scale = wopt / (2*(1+sor*sor));
    float second_scale = wopt*(sor*sor) / (2*(1+sor*sor));

    /// Macro to 2D Grid-Stride Loop
    PlaneThreads2D_GridStrideLoop {
        int in_idx = ix + 6 + iz * nx;
        int out_idx = in_idx;
        int stride = nx;

        float coefficient_1 = dx + dz;
        float coefficient_2 = 2*pow(dx,2)/2 + 2*pow(dz,2)/2;
        float coefficient_3 = 2*pow(dx,4)/24 + 2*pow(dz,4)/24;
        float coefficient_4 = 2*pow(dx,6)/720 + 2*pow(dz,6)/720;
        float coefficient_5 = 2*pow(dx,8)/40320 + 2*pow(dz,8)/40320;
        float coefficient_6 = 2*pow(dx,10)/3628800 + 2*pow(dz,10)/3628800;
        float coefficient_7 = 2*pow(dx,12)/479001600 + 2*pow(dz,12)/479001600;

        /// Load elemen from global memory to register
        float behind6 = d_pn[in_idx - 6 * stride];
        float behind5 = d_pn[in_idx - 5 * stride];
        float behind4 = d_pn[in_idx - 4 * stride];
        float behind3 = d_pn[in_idx - 3 * stride];
        float behind2 = d_pn[in_idx - 2 * stride];
        float behind1 = d_pn[in_idx - 1 * stride];
        float current = d_pn[in_idx];
        float infront1 = d_pn[in_idx + 1 * stride];
        float infront2 = d_pn[in_idx + 2 * stride];
        float infront3 = d_pn[in_idx + 3 * stride];
        float infront4 = d_pn[in_idx + 4 * stride];
        float infront5 = d_pn[in_idx + 5 * stride];
        float infront6 = d_pn[in_idx + 6 * stride];

        /// Shared memory
        __syncthreads();
        if (threadIdx.x < 6) {
            s_data[threadIdx.x] = d_pn[out_idx - 6];
            s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
        }
        s_data[threadIdx.x + 6] = current;
        __syncthreads();

        float value = ((first_scale+second_scale) * wopt) * (coefficient_1) * current;

        value += coefficient_2 * (first_scale * (infront1 + behind1) +
                                  second_scale * (s_data[threadIdx.x + 5] +
                                                  s_data[threadIdx.x + 7]));

        value += coefficient_3 * (first_scale * (infront2 + behind2) +
                                  second_scale * (s_data[threadIdx.x + 4] +
                                                  s_data[threadIdx.x + 8]));

        value += coefficient_4 * (first_scale * (infront3 + behind3) +
                                  second_scale * (s_data[threadIdx.x + 3] +
                                                  s_data[threadIdx.x + 9]));

        value += coefficient_5 * (first_scale * (infront4 + behind4) +
                                  second_scale * (s_data[threadIdx.x + 2] +
                                                  s_data[threadIdx.x + 10]));

        value += coefficient_6 * (first_scale * (infront5 + behind5) +
                                  second_scale * (s_data[threadIdx.x + 1] +
                                                  s_data[threadIdx.x + 11]));

        value += coefficient_7 * (first_scale * (infront6 + behind6) +
                                  second_scale * (s_data[threadIdx.x + 0] +
                                                  s_data[threadIdx.x + 12]));

        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("PSOR Optimized Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    PSOR_Optimized_time[ix] = elapsed_time;
}


__global__ void FVS_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iz = blockDim.y * blockIdx.y + threadIdx.y;
    float xscale_one, xscale_two, zscale_one, zscale_two;

    xscale_one = 0.5 * dt / dx;
    xscale_two = 0.5 * (dt*dt) / (dx*dx);

    zscale_one = 0.5 * dt / dz;
    zscale_two = 0.5 * (dt*dt) / (dz*dz);

    if (ix < nx-12 && iz < nz-12) {
        int in_idx = ix + 6;
        int out_idx = 0;
        int stride = nx;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        behind5 = d_pn[in_idx];
        in_idx += stride;
        behind4 = d_pn[in_idx];
        in_idx += stride;
        behind3 = d_pn[in_idx];
        in_idx += stride;
        behind2 = d_pn[in_idx];
        in_idx += stride;
        behind1 = d_pn[in_idx];
        in_idx += stride;

        current = d_pn[in_idx];
        out_idx = in_idx;
        in_idx += stride;

        infront1 = d_pn[in_idx];
        in_idx += stride;
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront3 = d_pn[in_idx];
        in_idx += stride;
        infront4 = d_pn[in_idx];
        in_idx += stride;
        infront5 = d_pn[in_idx];
        in_idx += stride;
        infront6 = d_pn[in_idx];
        in_idx += stride;

#pragma unroll
        for (iz = 6; iz < nz-6; iz++) {
            behind6 = behind5;
            behind5 = behind4;
            behind4 = behind3;
            behind3 = behind2;
            behind2 = behind1;
            behind1 = current;
            current = infront1;
            infront1 = infront2;
            infront2 = infront3;
            infront3 = infront4;
            infront4 = infront5;
            infront5 = infront6;
            infront6 = d_pn[in_idx];

            in_idx += stride;
            out_idx += stride;

            __syncthreads();

            if (threadIdx.x < 6) {
                s_data[threadIdx.x] = d_pn[out_idx - 6];
                s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale_one*dx + zscale_one*dz) +
                           (xscale_two*dx + zscale_two*dz)) * current;

            value += ((2 * pow(dz, 1) / 1) *
                        zscale_one * (infront1 - behind1) +
                      (2 * pow(dz, 2) / 2) *
                        zscale_two * (infront1 + behind1)) +
                     ((2 * pow(dx, 1) / 1) *
                        xscale_one * (s_data[threadIdx.x + 5] -
                                      s_data[threadIdx.x + 7]) +
                      (2 * pow(dx, 2) / 2) *
                        xscale_two * (s_data[threadIdx.x + 5] +
                                      s_data[threadIdx.x + 7]));

            value += ((2 * pow(dz, 3) / 6) *
                        zscale_one * (infront2 - behind2) +
                      (2 * pow(dz, 4) / 24) *
                        zscale_two * (infront2 + behind2)) +
                     ((2 * pow(dx, 3) / 6) *
                        xscale_one * (s_data[threadIdx.x + 4] -
                                      s_data[threadIdx.x + 8]) +
                      (2 * pow(dx, 4) / 24) *
                        xscale_two * (s_data[threadIdx.x + 4] +
                                      s_data[threadIdx.x + 8]));

            value += ((2 * pow(dz, 5) / 120) *
                        zscale_one * (infront3 - behind3) +
                      (2 * pow(dz, 6) / 720) *
                        zscale_two * (infront3 + behind3)) +
                     ((2 * pow(dx, 5) / 120) *
                        xscale_one * (s_data[threadIdx.x + 3] -
                                      s_data[threadIdx.x + 9]) +
                      (2 * pow(dx, 6) / 720) *
                        xscale_two * (s_data[threadIdx.x + 3] +
                                      s_data[threadIdx.x + 9]));

            value += ((2 * pow(dz, 7) / 5040) *
                        zscale_one * (infront4 - behind4) +
                      (2 * pow(dz, 8) / 40320) *
                        zscale_two * (infront4 + behind4)) +
                     ((2 * pow(dx, 7) / 5040) *
                        xscale_one * (s_data[threadIdx.x + 2] -
                                      s_data[threadIdx.x + 10]) +
                      (2 * pow(dx, 8) / 40320) *
                        xscale_two * (s_data[threadIdx.x + 2] +
                                      s_data[threadIdx.x + 10]));

            value += ((2 * pow(dz, 9) / 362880) *
                        zscale_one * (infront5 - behind5) +
                      (2 * pow(dz, 10) / 3628800) *
                        zscale_two * (infront5 + behind5)) +
                     ((2 * pow(dx, 9) / 362880) *
                        xscale_one * (s_data[threadIdx.x + 1] -
                                      s_data[threadIdx.x + 11]) +
                      (2 * pow(dx, 10) / 3628800) *
                        xscale_two * (s_data[threadIdx.x + 1] +
                                      s_data[threadIdx.x + 11]));

            value += ((2 * pow(dz, 11) / 39916800) *
                        zscale_one * (infront6 - behind6) +
                      (2 * pow(dz, 12) / 479001600) *
                        zscale_two * (infront6 + behind6)) +
                     ((2 * pow(dx, 11) / 39916800) *
                        xscale_one * (s_data[threadIdx.x + 0] -
                                      s_data[threadIdx.x + 12]) +
                      (2 * pow(dx, 12) / 479001600) *
                        xscale_two * (s_data[threadIdx.x + 0] +
                                      s_data[threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("FVS Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    FVS_time[ix] = elapsed_time;
}


/// Using Grid-Stride Loop
__global__ void FVS_2D_Solver_Optimized(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Get the start time
    clock_t start_time = clock();

    __shared__ float s_data[BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    /// Precompute factor scaling to reduce overhead of floating-point
    float xscale_one = 0.5 * dt / dx;
    float xscale_two = 0.5 * (dt*dt) / (dx*dx);

    float zscale_one = 0.5 * dt / dz;
    float zscale_two = 0.5 * (dt*dt) / (dz*dz);

    /// Macro to 2D Grid-Stride Loop
    PlaneThreads2D_GridStrideLoop {
        int in_idx = ix + 6 + iz * nx;
        int out_idx = in_idx;
        int stride = nx;

        /// Load elemen from global memory to register
        float behind6 = d_pn[in_idx - 6 * stride];
        float behind5 = d_pn[in_idx - 5 * stride];
        float behind4 = d_pn[in_idx - 4 * stride];
        float behind3 = d_pn[in_idx - 3 * stride];
        float behind2 = d_pn[in_idx - 2 * stride];
        float behind1 = d_pn[in_idx - 1 * stride];
        float current = d_pn[in_idx];
        float infront1 = d_pn[in_idx + 1 * stride];
        float infront2 = d_pn[in_idx + 2 * stride];
        float infront3 = d_pn[in_idx + 3 * stride];
        float infront4 = d_pn[in_idx + 4 * stride];
        float infront5 = d_pn[in_idx + 5 * stride];
        float infront6 = d_pn[in_idx + 6 * stride];

        /// Shared memory
        __syncthreads();
        if (threadIdx.x < 6) {
            s_data[threadIdx.x] = d_pn[out_idx - 6];
            s_data[threadIdx.x + BLOCK_DIMX + 6] = d_pn[out_idx + BLOCK_DIMX];
        }
        s_data[threadIdx.x + 6] = current;
        __syncthreads();

        float value = ((xscale_one*dx + zscale_one*dz) + (xscale_two*dx + zscale_two*dz)) * current;

        value += ((2 * pow(dz, 1) / 1) *
                    zscale_one * (infront1 - behind1) +
                  (2 * pow(dz, 2) / 2) *
                    zscale_two * (infront1 + behind1)) +
                 ((2 * pow(dx, 1) / 1) *
                    xscale_one * (s_data[threadIdx.x + 5] -
                                  s_data[threadIdx.x + 7]) +
                  (2 * pow(dx, 2) / 2) *
                    xscale_two * (s_data[threadIdx.x + 5] +
                                  s_data[threadIdx.x + 7]));

        value += ((2 * pow(dz, 3) / 6) *
                    zscale_one * (infront2 - behind2) +
                  (2 * pow(dz, 4) / 24) *
                    zscale_two * (infront2 + behind2)) +
                 ((2 * pow(dx, 3) / 6) *
                    xscale_one * (s_data[threadIdx.x + 4] -
                                  s_data[threadIdx.x + 8]) +
                  (2 * pow(dx, 4) / 24) *
                    xscale_two * (s_data[threadIdx.x + 4] +
                                  s_data[threadIdx.x + 8]));

        value += ((2 * pow(dz, 5) / 120) *
                    zscale_one * (infront3 - behind3) +
                  (2 * pow(dz, 6) / 720) *
                    zscale_two * (infront3 + behind3)) +
                 ((2 * pow(dx, 5) / 120) *
                    xscale_one * (s_data[threadIdx.x + 3] -
                                  s_data[threadIdx.x + 9]) +
                  (2 * pow(dx, 6) / 720) *
                    xscale_two * (s_data[threadIdx.x + 3] +
                                  s_data[threadIdx.x + 9]));

        value += ((2 * pow(dz, 7) / 5040) *
                    zscale_one * (infront4 - behind4) +
                  (2 * pow(dz, 8) / 40320) *
                    zscale_two * (infront4 + behind4)) +
                 ((2 * pow(dx, 7) / 5040) *
                    xscale_one * (s_data[threadIdx.x + 2] -
                                  s_data[threadIdx.x + 10]) +
                  (2 * pow(dx, 8) / 40320) *
                    xscale_two * (s_data[threadIdx.x + 2] +
                                  s_data[threadIdx.x + 10]));

        value += ((2 * pow(dz, 9) / 362880) *
                    zscale_one * (infront5 - behind5) +
                  (2 * pow(dz, 10) / 3628800) *
                    zscale_two * (infront5 + behind5)) +
                 ((2 * pow(dx, 9) / 362880) *
                    xscale_one * (s_data[threadIdx.x + 1] -
                                  s_data[threadIdx.x + 11]) +
                  (2 * pow(dx, 10) / 3628800) *
                    xscale_two * (s_data[threadIdx.x + 1] +
                                  s_data[threadIdx.x + 11]));

        value += ((2 * pow(dz, 11) / 39916800) *
                    zscale_one * (infront6 - behind6) +
                  (2 * pow(dz, 12) / 479001600) *
                    zscale_two * (infront6 + behind6)) +
                 ((2 * pow(dx, 11) / 39916800) *
                    xscale_one * (s_data[threadIdx.x + 0] -
                                  s_data[threadIdx.x + 12]) +
                  (2 * pow(dx, 12) / 479001600) *
                    xscale_two * (s_data[threadIdx.x + 0] +
                                  s_data[threadIdx.x + 12]));

        d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
    }

    /// Get the end time
    clock_t end_time = clock();

    /// Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    /// Print the elapsed time
    /// printf("FVS Optimized Execution time: %f ms\n", elapsed_time);

    /// Save the elapsed time to the global array
    FVS_Optimized_time[ix] = elapsed_time;
}


void Measure_And_Execute_Kernels(
    int nx, int nz, float dx, float dz, float dt, float *d_v,
    float *d_pn_Galerkin, float *d_pp_Galerkin,
    float *d_pn_Galerkin_Optimized, float *d_pp_Galerkin_Optimized,
    float *d_pn_Leapfrog, float *d_pp_Leapfrog,
    float *d_pn_Leapfrog_Optimized, float *d_pp_Leapfrog_Optimized,
    float *d_pn_CrankNicolson, float *d_pp_CrankNicolson,
    float *d_pn_CrankNicolson_Optimized, float *d_pp_CrankNicolson_Optimized,
    float *d_pn_ADI, float *d_pp_ADI,
    float *d_pn_ADI_Optimized, float *d_pp_ADI_Optimized,
    float *d_pn_Sigma, float *d_pp_Sigma,
    float *d_pn_Sigma_Optimized, float *d_pp_Sigma_Optimized,
    float *d_pn_LaxWendroff, float *d_pp_LaxWendroff,
    float *d_pn_LaxWendroff_Optimized, float *d_pp_LaxWendroff_Optimized,
    float *d_pn_FractionalStep, float *d_pp_FractionalStep,
    float *d_pn_FractionalStep_Optimized, float *d_pp_FractionalStep_Optimized,
    float *d_pn_MacCormack, float *d_pp_MacCormack,
    float *d_pn_MacCormack_Optimized, float *d_pp_MacCormack_Optimized,
    float *d_pn_TVD, float *d_pp_TVD,
    float *d_pn_TVD_Optimized, float *d_pp_TVD_Optimized,
    float *d_pn_PSOR, float *d_pp_PSOR,
    float *d_pn_PSOR_Optimized, float *d_pp_PSOR_Optimized,
    float *d_pn_FVS, float *d_pp_FVS,
    float *d_pn_FVS_Optimized, float *d_pp_FVS_Optimized,
    float *h_pn_Galerkin, float *h_pp_Galerkin,
    float *h_pn_Galerkin_Optimized, float *h_pp_Galerkin_Optimized,
    float *h_pn_Leapfrog, float *h_pp_Leapfrog,
    float *h_pn_Leapfrog_Optimized, float *h_pp_Leapfrog_Optimized,
    float *h_pn_CrankNicolson, float *h_pp_CrankNicolson,
    float *h_pn_CrankNicolson_Optimized, float *h_pp_CrankNicolson_Optimized,
    float *h_pn_ADI, float *h_pp_ADI,
    float *h_pn_ADI_Optimized, float *h_pp_ADI_Optimized,
    float *h_pn_Sigma, float *h_pp_Sigma,
    float *h_pn_Sigma_Optimized, float *h_pp_Sigma_Optimized,
    float *h_pn_LaxWendroff, float *h_pp_LaxWendroff,
    float *h_pn_LaxWendroff_Optimized, float *h_pp_LaxWendroff_Optimized,
    float *h_pn_FractionalStep, float *h_pp_FractionalStep,
    float *h_pn_FractionalStep_Optimized, float *h_pp_FractionalStep_Optimized,
    float *h_pn_MacCormack, float *h_pp_MacCormack,
    float *h_pn_MacCormack_Optimized, float *h_pp_MacCormack_Optimized,
    float *h_pn_TVD, float *h_pp_TVD,
    float *h_pn_TVD_Optimized, float *h_pp_TVD_Optimized,
    float *h_pn_PSOR, float *h_pp_PSOR,
    float *h_pn_PSOR_Optimized, float *h_pp_PSOR_Optimized,
    float *h_pn_FVS, float *h_pp_FVS,
    float *h_pn_FVS_Optimized, float *h_pp_FVS_Optimized
)
{
    int num_gpus = 0;
    int MAX_GPU = 8;
    int NUM_EXECUTIONS = 50; /// Number of kernel execution

    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        fprintf(stderr, "No GPU available!\n");
        return;
    }
    num_gpus = (num_gpus > MAX_GPU) ? MAX_GPU : num_gpus;

    int remainder = nz % num_gpus;
    std::vector<cudaStream_t> streams(num_gpus);

    struct KernelData {
        float *d_pn;
        float *d_pp;
        float *h_pn;
        float *h_pp;
        const char *name;
        const char *time_file;
        const char *result_file;
    };

    KernelData kernelData[] = {
        {d_pn_Galerkin, d_pp_Galerkin, h_pp_Galerkin, h_pn_Galerkin, "2D Galerkin Kernel", "GalerkinTime_2D_data.txt", "GalerkinResults_2D_data.bin"},
        {d_pn_Galerkin_Optimized, d_pp_Galerkin_Optimized, h_pp_Galerkin_Optimized, h_pn_Galerkin_Optimized, "2D Galerkin Optimized Kernel", "GalerkinOptimizedTime_2D_data.txt", "GalerkinOptimizedResults_2D_data.bin"},
        {d_pn_Leapfrog, d_pp_Leapfrog, h_pp_Leapfrog, h_pn_Leapfrog, "2D Leapfrog Kernel", "LeapfrogTime_2D_data.txt", "LeapfrogResults_2D_data.bin"},
        {d_pn_Leapfrog_Optimized, d_pp_Leapfrog_Optimized, h_pp_Leapfrog_Optimized, h_pn_Leapfrog_Optimized, "2D Leapfrog Optimized Kernel", "LeapfrogOptimizedTime_2D_data.txt", "LeapfrogOptimizedResults_2D_data.bin"},
        {d_pn_CrankNicolson, d_pp_CrankNicolson, h_pp_CrankNicolson, h_pn_CrankNicolson, "2D Crank-Nicolson Kernel", "CrankNicolsonTime_2D_data.txt", "CrankNicolsonResults_2D_data.bin"},
        {d_pn_CrankNicolson_Optimized, d_pp_CrankNicolson_Optimized, h_pp_CrankNicolson_Optimized, h_pn_CrankNicolson_Optimized, "2D Crank-Nicolson Optimized Kernel", "CrankNicolsonOptimizedTime_2D_data.txt", "CrankNicolsonOptimizedResults_2D_data.bin"},
        {d_pn_ADI, d_pp_ADI, h_pp_ADI, h_pn_ADI, "2D ADI Kernel", "ADITime_2D_data.txt", "ADIResults_2D_data.bin"},
        {d_pn_ADI_Optimized, d_pp_ADI_Optimized, h_pp_ADI_Optimized, h_pn_ADI_Optimized, "2D ADI Optimized Kernel", "ADIOptimizedTime_2D_data.txt", "ADIOptimizedResults_2D_data.bin"},
        {d_pn_Sigma, d_pp_Sigma, h_pp_Sigma, h_pn_Sigma, "2D Sigma Kernel", "SigmaTime_2D_data.txt", "SigmaResults_2D_data.bin"},
        {d_pn_Sigma_Optimized, d_pp_Sigma_Optimized, h_pp_Sigma_Optimized, h_pn_Sigma_Optimized, "2D Sigma Optimized Kernel", "SigmaOptimizedTime_2D_data.txt", "SigmaOptimizedResults_2D_data.bin"},
        {d_pn_LaxWendroff, d_pp_LaxWendroff, h_pp_LaxWendroff, h_pn_LaxWendroff, "2D Lax-Wendroff Kernel", "LaxWendroffTime_2D_data.txt", "LaxWendroffResults_2D_data.bin"},
        {d_pn_LaxWendroff_Optimized, d_pp_LaxWendroff_Optimized, h_pp_LaxWendroff_Optimized, h_pn_LaxWendroff_Optimized, "2D Lax-Wendroff Optimized Kernel", "LaxWendroffOptimizedTime_2D_data.txt", "LaxWendroffOptimizedResults_2D_data.bin"},
        {d_pn_FractionalStep, d_pp_FractionalStep, h_pp_FractionalStep, h_pn_FractionalStep, "2D Fractional Step Kernel", "FractionalStepTime_2D_data.txt", "FractionalStepResults_2D_data.bin"},
        {d_pn_FractionalStep_Optimized, d_pp_FractionalStep_Optimized, h_pp_FractionalStep_Optimized, h_pn_FractionalStep_Optimized, "2D Fractional Step Optimized Kernel", "FractionalStepOptimizedTime_2D_data.txt", "FractionalStepOptimizedResults_2D_data.bin"},
        {d_pn_MacCormack, d_pp_MacCormack, h_pp_MacCormack, h_pn_MacCormack, "2D MacCormack Kernel", "MacCormackTime_2D_data.txt", "MacCormackResults_2D_data.bin"},
        {d_pn_MacCormack_Optimized, d_pp_MacCormack_Optimized, h_pp_MacCormack_Optimized, h_pn_MacCormack_Optimized, "2D MacCormack Optimized Kernel", "MacCormackOptimizedTime_2D_data.txt", "MacCormackOptimizedResults_2D_data.bin"},
        {d_pn_TVD, d_pp_TVD, h_pp_TVD, h_pn_TVD, "2D TVD Kernel", "TVDTime_2D_data.txt", "TVDResults_2D_data.bin"},
        {d_pn_TVD_Optimized, d_pp_TVD_Optimized, h_pp_TVD_Optimized, h_pn_TVD_Optimized, "2D TVD Optimized Kernel", "TVDOptimizedTime_2D_data.txt", "TVDOptimizedResults_2D_data.bin"},
        {d_pn_PSOR, d_pp_PSOR, h_pp_PSOR, h_pn_PSOR, "2D PSOR Kernel", "PSORTime_2D_data.txt", "PSORResults_2D_data.bin"},
        {d_pn_PSOR_Optimized, d_pp_PSOR_Optimized, h_pp_PSOR_Optimized, h_pn_PSOR_Optimized, "2D PSOR Optimized Kernel", "PSOROptimizedTime_2D_data.txt", "PSOROptimizedResults_2D_data.bin"},
        {d_pn_FVS, d_pp_FVS, h_pp_FVS, h_pn_FVS, "2D FVS Kernel", "FVSTime_2D_data.txt", "FVSResults_2D_data.bin"},
        {d_pn_FVS_Optimized, d_pp_FVS_Optimized, h_pp_FVS_Optimized, h_pn_FVS_Optimized, "2D FVS Optimized Kernel", "FVSOptimizedTime_2D_data.txt", "FVSOptimizedResults_2D_data.bin"},
    };

    /// Number of kernel based on kernelData[]
    int number_of_kernels = sizeof(kernelData) / sizeof(kernelData[0]);

    /// Array to save execution time data
    float executionTimes[number_of_kernels][MAX_GPU][NUM_EXECUTIONS] = {0};

#pragma unroll
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        int chunk_size = (nz / num_gpus) + (gpu < remainder ? 1 : 0);
        int z_offset = (gpu < remainder) ? gpu * chunk_size : gpu * (nz / num_gpus) + remainder;
        float *d_v_chunk = d_v + z_offset * nx;

        cudaStreamCreate(&streams[gpu]);

#pragma unroll
        for (int i = 0; i < number_of_kernels; i++) {
#pragma unroll
            for (int iter = 0; iter < NUM_EXECUTIONS; iter++) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaDeviceSynchronize();
                cudaEventRecord(start, 0);

                dim3 blockDim(8, 8);
                dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                            (chunk_size + blockDim.y - 1) / blockDim.y);

                dim3 blockSize(8, 8);
                dim3 gridSize((nx + blockSize.x - 1) / blockSize.x);

                switch (i) {
                    case 0:
                        Galerkin_2D_Solver<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 1:
                        Galerkin_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 2:
                        Leapfrog_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 3:
                        Leapfrog_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 4:
                        CrankNicolson_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 5:
                        CrankNicolson_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 6:
                        ADI_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 7:
                        ADI_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 8:
                        Sigma_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 9:
                        Sigma_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 10:
                        LaxWendroff_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 11:
                        LaxWendroff_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 12:
                        FractionalStep_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 13:
                        FractionalStep_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 14:
                        MacCormack_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 15:
                        MacCormack_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 16:
                        TVD_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 17:
                        TVD_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 18:
                        PSOR_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 19:
                        PSOR_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 20:
                        FVS_2D_Solver<<<gridSize, blockSize, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 21:
                        FVS_2D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                }

                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaDeviceSynchronize();

                float timeElapsed = 0;
                cudaEventElapsedTime(&timeElapsed, start, stop);
                executionTimes[i][gpu][iter] = timeElapsed;
                //printf("GPU %d: %s execution time: %f ms\n", gpu, kernelData[i].name, timeElapsed);

                cudaMemcpy(kernelData[i].h_pp, kernelData[i].d_pp, nx * chunk_size * sizeof(float), cudaMemcpyDeviceToHost);

                /*
                FILE *resultFile = fopen(kernelData[i].result_file, "w");
                if (resultFile == NULL) {
                    fprintf(stderr, "Error opening file %s\n", kernelData[i].result_file);
                }
                else {
                    for (int j = 0; j < nx * ny * chunk_size; j++) {
                        fprintf(resultFile, "%f\n", kernelData[i].h_pp[j]);
                    }
                    fclose(resultFile);
                }
                */

                FILE *resultFile = fopen(kernelData[i].result_file, "wb");
                if (resultFile) {
                    fwrite(kernelData[i].h_pp, sizeof(float), nx  * chunk_size, resultFile);
                    fclose(resultFile);
                }

                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
        }
    }

#pragma unroll
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaStreamSynchronize(streams[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }

    time_t now = time(NULL);
    char *timestamp = ctime(&now);
    timestamp[strlen(timestamp) - 1] = '\0';

#pragma unroll
    for (int i = 0; i < number_of_kernels; i++) {
        FILE *file = fopen(kernelData[i].time_file, "a");
        if (file == NULL) {
            fprintf(stderr, "Error opening file %s\n", kernelData[i].time_file);
            continue;
        }

        fprintf(file, "===========================================\n");
        fprintf(file, "Execution Log: %s\n", timestamp);
        fprintf(file, "Execution times (ms) for %s:\n", kernelData[i].name);

#pragma unroll
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            float sumTime = 0;
            fprintf(file, "GPU %d:\n", gpu);
#pragma unroll
            for (int iter = 0; iter < NUM_EXECUTIONS; iter++) {
                fprintf(file, "  Iteration %d: %f ms\n", iter + 1, executionTimes[i][gpu][iter]);
                sumTime += executionTimes[i][gpu][iter];
            }
            float avgTime = sumTime / NUM_EXECUTIONS;
            fprintf(file, "  Average Time: %f ms\n", avgTime);
        }

        fprintf(file, "===========================================\n\n");
        fclose(file);
    }
}



int main (int argc, char **argv) {
    /// Set problem size
    int nx = NX;
    int nz = NZ;
    int size = nx * nz;

    /// Set simulation parameters
    float dx = DX;
    float dz = DZ;
    float dt = DT;

    /// Allocate memory on the host
    float *h_pn_Galerkin = (float*)malloc(size * sizeof(float));
    float *h_pn_Galerkin_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_Leapfrog = (float*)malloc(size * sizeof(float));
    float *h_pn_Leapfrog_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_CrankNicolson = (float*)malloc(size * sizeof(float));
    float *h_pn_CrankNicolson_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_ADI = (float*)malloc(size * sizeof(float));
    float *h_pn_ADI_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_Sigma = (float*)malloc(size * sizeof(float));
    float *h_pn_Sigma_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_LaxWendroff = (float*)malloc(size * sizeof(float));
    float *h_pn_LaxWendroff_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_FractionalStep = (float*)malloc(size * sizeof(float));
    float *h_pn_FractionalStep_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_MacCormack = (float*)malloc(size * sizeof(float));
    float *h_pn_MacCormack_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_TVD = (float*)malloc(size * sizeof(float));
    float *h_pn_TVD_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_PSOR = (float*)malloc(size * sizeof(float));
    float *h_pn_PSOR_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pn_FVS = (float*)malloc(size * sizeof(float));
    float *h_pn_FVS_Optimized = (float*)malloc(size * sizeof(float));
    float *h_v = (float*)malloc(size * sizeof(float));
    float *h_pp_Galerkin = (float*)malloc(size * sizeof(float));
    float *h_pp_Galerkin_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_Leapfrog = (float*)malloc(size * sizeof(float));
    float *h_pp_Leapfrog_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_CrankNicolson = (float*)malloc(size * sizeof(float));
    float *h_pp_CrankNicolson_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_ADI = (float*)malloc(size * sizeof(float));
    float *h_pp_ADI_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_Sigma = (float*)malloc(size * sizeof(float));
    float *h_pp_Sigma_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_LaxWendroff = (float*)malloc(size * sizeof(float));
    float *h_pp_LaxWendroff_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_FractionalStep = (float*)malloc(size * sizeof(float));
    float *h_pp_FractionalStep_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_MacCormack = (float*)malloc(size * sizeof(float));
    float *h_pp_MacCormack_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_TVD = (float*)malloc(size * sizeof(float));
    float *h_pp_TVD_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_PSOR = (float*)malloc(size * sizeof(float));
    float *h_pp_PSOR_Optimized = (float*)malloc(size * sizeof(float));
    float *h_pp_FVS = (float*)malloc(size * sizeof(float));
    float *h_pp_FVS_Optimized = (float*)malloc(size * sizeof(float));

    /// Initialize input data with random values
#pragma unroll
    for (int i=0; i < size; i++) {
	    h_pn_Galerkin[i] = h_pn_Galerkin_Optimized[i] =
	    h_pn_Leapfrog[i] = h_pn_Leapfrog_Optimized[i] =
	    h_pn_CrankNicolson[i] = h_pn_CrankNicolson_Optimized[i] =
	    h_pn_ADI[i] = h_pn_ADI_Optimized[i] =
	    h_pn_Sigma[i] = h_pn_Sigma_Optimized[i] =
	    h_pn_LaxWendroff[i] = h_pn_LaxWendroff_Optimized[i] =
	    h_pn_FractionalStep[i] = h_pn_FractionalStep_Optimized[i] =
        h_pn_TVD[i] = h_pn_TVD_Optimized[i] =
        h_pn_MacCormack[i] = h_pn_MacCormack_Optimized[i] =
        h_pn_PSOR[i] = h_pn_PSOR_Optimized[i] =
	    h_pn_FVS[i] = h_pn_FVS_Optimized[i] = 1 + rand() % 1000;
    }

    /// Pointer to allocate memory on the device
    float *d_pn_Galerkin, *d_pn_Galerkin_Optimized;
    float *d_pn_Leapfrog, *d_pn_Leapfrog_Optimized;
    float *d_pn_CrankNicolson, *d_pn_CrankNicolson_Optimized;
    float *d_pn_ADI, *d_pn_ADI_Optimized;
    float *d_pn_Sigma, *d_pn_Sigma_Optimized;
    float *d_pn_LaxWendroff, *d_pn_LaxWendroff_Optimized;
    float *d_pn_FractionalStep, *d_pn_FractionalStep_Optimized;
    float *d_pn_MacCormack, *d_pn_MacCormack_Optimized;
    float *d_pn_TVD, *d_pn_TVD_Optimized;
    float *d_pn_PSOR, *d_pn_PSOR_Optimized;
    float *d_pn_FVS, *d_pn_FVS_Optimized;
    float *d_v;
    float *d_pp_Galerkin, *d_pp_Galerkin_Optimized;
    float *d_pp_Leapfrog, *d_pp_Leapfrog_Optimized;
    float *d_pp_CrankNicolson, *d_pp_CrankNicolson_Optimized;
    float *d_pp_ADI, *d_pp_ADI_Optimized;
    float *d_pp_Sigma, *d_pp_Sigma_Optimized;
    float *d_pp_LaxWendroff, *d_pp_LaxWendroff_Optimized;
    float *d_pp_FractionalStep, *d_pp_FractionalStep_Optimized;
    float *d_pp_MacCormack, *d_pp_MacCormack_Optimized;
    float *d_pp_TVD, *d_pp_TVD_Optimized;
    float *d_pp_PSOR, *d_pp_PSOR_Optimized;
    float *d_pp_FVS, *d_pp_FVS_Optimized;

    /// Allocate memory on the device
    cudaMalloc((void**)&d_pn_Galerkin, size * sizeof(float));
    cudaMalloc((void**)&d_pn_Galerkin_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_Leapfrog, size * sizeof(float));
    cudaMalloc((void**)&d_pn_Leapfrog_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_CrankNicolson, size * sizeof(float));
    cudaMalloc((void**)&d_pn_CrankNicolson_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_ADI, size * sizeof(float));
    cudaMalloc((void**)&d_pn_ADI_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_Sigma, size * sizeof(float));
    cudaMalloc((void**)&d_pn_Sigma_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_LaxWendroff, size * sizeof(float));
    cudaMalloc((void**)&d_pn_LaxWendroff_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_FractionalStep, size * sizeof(float));
    cudaMalloc((void**)&d_pn_FractionalStep_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_MacCormack, size * sizeof(float));
    cudaMalloc((void**)&d_pn_MacCormack_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_TVD, size * sizeof(float));
    cudaMalloc((void**)&d_pn_TVD_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_PSOR, size * sizeof(float));
    cudaMalloc((void**)&d_pn_PSOR_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pn_FVS, size * sizeof(float));
    cudaMalloc((void**)&d_pn_FVS_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_v, size * sizeof(float));
    cudaMalloc((void**)&d_pp_Galerkin, size * sizeof(float));
    cudaMalloc((void**)&d_pp_Galerkin_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_Leapfrog, size * sizeof(float));
    cudaMalloc((void**)&d_pp_Leapfrog_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_CrankNicolson, size * sizeof(float));
    cudaMalloc((void**)&d_pp_CrankNicolson_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_ADI, size * sizeof(float));
    cudaMalloc((void**)&d_pp_ADI_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_Sigma, size * sizeof(float));
    cudaMalloc((void**)&d_pp_Sigma_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_LaxWendroff, size * sizeof(float));
    cudaMalloc((void**)&d_pp_LaxWendroff_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_FractionalStep, size * sizeof(float));
    cudaMalloc((void**)&d_pp_FractionalStep_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_MacCormack, size * sizeof(float));
    cudaMalloc((void**)&d_pp_MacCormack_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_TVD, size * sizeof(float));
    cudaMalloc((void**)&d_pp_TVD_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_PSOR, size * sizeof(float));
    cudaMalloc((void**)&d_pp_PSOR_Optimized, size * sizeof(float));
    cudaMalloc((void**)&d_pp_FVS, size * sizeof(float));
    cudaMalloc((void**)&d_pp_FVS_Optimized, size * sizeof(float));

    /// Transfer input data from host to device
    cudaMemcpy(d_pn_Galerkin, h_pn_Galerkin, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_Leapfrog, h_pn_Leapfrog, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_Leapfrog_Optimized, h_pn_Leapfrog_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_CrankNicolson, h_pn_CrankNicolson, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_CrankNicolson_Optimized, h_pn_CrankNicolson_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_ADI, h_pn_ADI, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_ADI_Optimized, h_pn_ADI_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_Sigma, h_pn_Sigma, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_Sigma_Optimized, h_pn_Sigma_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_LaxWendroff, h_pn_LaxWendroff, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_LaxWendroff_Optimized, h_pn_LaxWendroff_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_FractionalStep, h_pn_FractionalStep, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_FractionalStep_Optimized, h_pn_FractionalStep_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_MacCormack, h_pn_MacCormack, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_MacCormack_Optimized, h_pn_MacCormack_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_TVD, h_pn_TVD, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_TVD_Optimized, h_pn_TVD_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_PSOR, h_pn_PSOR, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_PSOR_Optimized, h_pn_PSOR_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_FVS, h_pn_FVS, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_FVS_Optimized, h_pn_FVS_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);

    /// Define grid and block dimensions
    dim3 blockSize(8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x);

    dim3 blockSize2(8);
    dim3 gridSize2((nx + blockSize2.x - 1) / blockSize2.x);

    /// Current block size
    /// int currentBlockSize = blockSize.x * blockSize.y * blockSize.z;
    int blockSizeLimit;
    cudaDeviceGetAttribute(&blockSizeLimit, cudaDevAttrMaxThreadsPerBlock,0);
    printf("Max Threads Per Block: %d\n", blockSizeLimit);


    ///=============================================================
    /// Launch Galerkin Kernel and measure time
    ///=============================================================
/*
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    Galerkin_2D_Solver<<<gridSize, blockSize>>>(nx, dx, nz, dz, dt, d_v, d_pn_Galerkin, d_pp_Galerkin);
    checkCUDAError("2D Galerkin Kernel launch");

    cudaEventRecord(stop1);
    cudaDeviceSynchronize();

    float time1 = 0;
    cudaEventElapsedTime(&time1, start1, stop1);
    printf("Total Execution Time on GPU for 2D Galerkin kernel: %f ms\n", time1);

    /// Transfer the result of Discontinuous_Galerkin_2D_Solver from device to host
    cudaMemcpy(h_pp_Galerkin, d_pp_Galerkin, size*sizeof(float),cudaMemcpyDeviceToHost);

    /// Transfer the Galerkin_time array from device to host
    float GalerkinTime[N];
    cudaMemcpyFromSymbol(GalerkinTime, Galerkin_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    /// Save the result of Galerkin elapsed time to a file1
    FILE *file1 = fopen("GalerkinTime_2D_data.txt", "w");
    if (file1 == NULL) {
    	fprintf(stderr, "Error opening GalerkinTime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file1, "%.6f\n", GalerkinTime[i]);
    }
    fclose(file1);
/*
    /// Save the result of Galerkin_2D_solver to a file_a
    FILE *file_a = fopen("Galerkin2DSolver.txt", "w");
    if (file_a == NULL) {
	    fprintf(stderr, "Error opening Galerkin2DSolver.txt file..\n");
	    return 1;
    }
    for (int j=0; j<size; j++) {
	    fprintf(file_a, "%.6f\n", h_pp_Galerkin[j]);
    }
    fclose(file_a);
*/

    Measure_And_Execute_Kernels(nx, nz, dx, dz, dt, d_v,
        d_pn_Galerkin, d_pp_Galerkin, d_pn_Galerkin_Optimized, d_pp_Galerkin_Optimized,
        d_pn_Leapfrog, d_pp_Leapfrog, d_pn_Leapfrog_Optimized, d_pp_Leapfrog_Optimized,
        d_pn_CrankNicolson, d_pp_CrankNicolson, d_pn_CrankNicolson_Optimized, d_pp_CrankNicolson_Optimized,
        d_pn_ADI, d_pp_ADI, d_pn_ADI_Optimized, d_pp_ADI_Optimized,
        d_pn_Sigma, d_pp_Sigma, d_pn_Sigma_Optimized, d_pp_Sigma_Optimized,
        d_pn_LaxWendroff, d_pp_LaxWendroff, d_pn_LaxWendroff_Optimized, d_pp_LaxWendroff_Optimized,
        d_pn_FractionalStep, d_pp_FractionalStep, d_pn_FractionalStep_Optimized, d_pp_FractionalStep_Optimized,
        d_pn_MacCormack, d_pp_MacCormack, d_pn_MacCormack_Optimized, d_pp_MacCormack_Optimized,
        d_pn_TVD, d_pp_TVD, d_pn_TVD_Optimized, d_pp_TVD_Optimized,
        d_pn_PSOR, d_pp_PSOR, d_pn_PSOR_Optimized, d_pp_PSOR_Optimized,
        d_pn_FVS, d_pp_FVS, d_pn_FVS_Optimized, d_pp_FVS_Optimized,
        h_pn_Galerkin, h_pp_Galerkin, h_pn_Galerkin_Optimized, h_pp_Galerkin_Optimized,
        h_pn_Leapfrog, h_pp_Leapfrog, h_pn_Leapfrog_Optimized, h_pp_Leapfrog_Optimized,
        h_pn_CrankNicolson, h_pp_CrankNicolson, h_pn_CrankNicolson_Optimized, h_pp_CrankNicolson_Optimized,
        h_pn_ADI, h_pp_ADI, h_pn_ADI_Optimized, h_pp_ADI_Optimized,
        h_pn_Sigma, h_pp_Sigma, h_pn_Sigma_Optimized, h_pp_Sigma_Optimized,
        h_pn_LaxWendroff,  h_pp_LaxWendroff, h_pn_LaxWendroff_Optimized,  h_pp_LaxWendroff_Optimized,
        h_pn_FractionalStep, h_pp_FractionalStep, h_pn_FractionalStep_Optimized, h_pp_FractionalStep_Optimized,
        h_pn_MacCormack, h_pp_MacCormack, h_pn_MacCormack_Optimized, h_pp_MacCormack_Optimized,
        h_pn_TVD, h_pp_TVD, h_pn_TVD_Optimized, h_pp_TVD_Optimized,
        h_pn_PSOR, h_pp_PSOR, h_pn_PSOR_Optimized, h_pp_PSOR_Optimized,
        h_pn_FVS, h_pp_FVS, h_pn_FVS_Optimized, h_pp_FVS_Optimized
    );

/*
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
*/

    free(h_pn_Galerkin);
    free(h_pn_Galerkin_Optimized);
    free(h_pn_Leapfrog);
    free(h_pn_Leapfrog_Optimized);
    free(h_pn_CrankNicolson);
    free(h_pn_CrankNicolson_Optimized);
    free(h_pn_ADI);
    free(h_pn_ADI_Optimized);
    free(h_pn_Sigma);
    free(h_pn_Sigma_Optimized);
    free(h_pn_LaxWendroff);
    free(h_pn_LaxWendroff_Optimized);
    free(h_pn_FractionalStep);
    free(h_pn_FractionalStep_Optimized);
    free(h_pn_MacCormack);
    free(h_pn_MacCormack_Optimized);
    free(h_pn_TVD);
    free(h_pn_TVD_Optimized);
    free(h_pn_PSOR);
    free(h_pn_PSOR_Optimized);
    free(h_pn_FVS);
    free(h_pn_FVS_Optimized);
    free(h_v);
    free(h_pp_Galerkin);
    free(h_pp_Galerkin_Optimized);
    free(h_pp_Leapfrog);
    free(h_pp_Leapfrog_Optimized);
    free(h_pp_CrankNicolson);
    free(h_pp_CrankNicolson_Optimized);
    free(h_pp_ADI);
    free(h_pp_ADI_Optimized);
    free(h_pp_Sigma);
    free(h_pp_Sigma_Optimized);
    free(h_pp_LaxWendroff);
    free(h_pp_LaxWendroff_Optimized);
    free(h_pp_FractionalStep);
    free(h_pp_FractionalStep_Optimized);
    free(h_pp_MacCormack);
    free(h_pp_MacCormack_Optimized);
    free(h_pp_TVD);
    free(h_pp_TVD_Optimized);
    free(h_pp_PSOR);
    free(h_pp_PSOR_Optimized);
    free(h_pp_FVS);
    free(h_pp_FVS_Optimized);

    //free(GalerkinTime);
    //free(LeapfrogTime);
    //free(TVDTime);

    cudaFree(d_pn_Galerkin);
    cudaFree(d_pn_Galerkin_Optimized);
    cudaFree(d_pn_Leapfrog);
    cudaFree(d_pn_Leapfrog_Optimized);
    cudaFree(d_pn_CrankNicolson);
    cudaFree(d_pn_CrankNicolson_Optimized);
    cudaFree(d_pn_ADI);
    cudaFree(d_pn_ADI_Optimized);
    cudaFree(d_pn_Sigma);
    cudaFree(d_pn_Sigma_Optimized);
    cudaFree(d_pn_LaxWendroff);
    cudaFree(d_pn_LaxWendroff_Optimized);
    cudaFree(d_pn_FractionalStep);
    cudaFree(d_pn_FractionalStep_Optimized);
    cudaFree(d_pn_MacCormack);
    cudaFree(d_pn_MacCormack_Optimized);
    cudaFree(d_pn_TVD);
    cudaFree(d_pn_TVD_Optimized);
    cudaFree(d_pn_PSOR);
    cudaFree(d_pn_PSOR_Optimized);
    cudaFree(d_pn_FVS);
    cudaFree(d_pn_FVS_Optimized);
    cudaFree(d_v);
    cudaFree(d_pp_Galerkin);
    cudaFree(d_pp_Galerkin_Optimized);
    cudaFree(d_pp_Leapfrog);
    cudaFree(d_pp_Leapfrog_Optimized);
    cudaFree(d_pp_CrankNicolson);
    cudaFree(d_pp_CrankNicolson_Optimized);
    cudaFree(d_pp_ADI);
    cudaFree(d_pp_ADI_Optimized);
    cudaFree(d_pp_Sigma);
    cudaFree(d_pp_Sigma_Optimized);
    cudaFree(d_pp_LaxWendroff);
    cudaFree(d_pp_LaxWendroff_Optimized);
    cudaFree(d_pp_FractionalStep);
    cudaFree(d_pp_FractionalStep_Optimized);
    cudaFree(d_pp_MacCormack);
    cudaFree(d_pp_MacCormack_Optimized);
    cudaFree(d_pp_TVD);
    cudaFree(d_pp_TVD_Optimized);
    cudaFree(d_pp_PSOR);
    cudaFree(d_pp_PSOR_Optimized);
    cudaFree(d_pp_FVS);
    cudaFree(d_pp_FVS_Optimized);

    return 0;
}






