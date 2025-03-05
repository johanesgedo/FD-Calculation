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
//#include <cutensor.h>
//#include <cuquantum.h>

#define NX 256
#define NY 256
#define NZ 256

#define DX 12.5
#define DY 12.5
#define DZ 12.5
#define DT 0.1

#define N 1024

#define BLOCK_DIMX 8
#define BLOCK_DIMY 8
#define BLOCK_DIMZ 8
#define sigma1  0.25
#define sigma2  0.75
#define IM  4.0f
#define JM  4.0f

#define p_TM 8
#define p_NF 8

#define cubeThreads 						\
    for (int k = threadIdx.z; k < p_TM; k += blockDim.z) 	\
        for (int j = threadIdx.y; j < p_TM; j += blockDim.y) 	\
            for (int i = threadIdx.x; i < p_TM; i += blockDim.x)

/// Combination of Grid-Stride Loop and OCCA (Optimized Combinatorical Construction using Algorithm)
#define cubeThreads_GridStrideLoop \
    for (int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < NX-12; ix += gridDim.x * blockDim.x) \
        for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < NY-12; iy += gridDim.y * blockDim.y) \
            for (int iz = blockIdx.z * blockDim.z + threadIdx.z; iz < NZ-12; iz += gridDim.z * blockDim.z)

void checkCUDAError(const char *message) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
    	fprintf(stderr, "CUDA Error: %s: %s.\n", message, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
}


/// Discontinuous Galerkin method to solve 3D acoustic wave equation using OCCA algorithm
__global__ void Discontinuous_Galerkin_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    /// Ensure that threads are within the grid size
    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;

        /// Thread-local input and output arrays
        __shared__ float r_pn[p_NF]; // thread-local input
        __shared__ float r_pp[p_NF]; // thread-local output

        /// Shared memory arrays for second derivatives
        __shared__ float s_d2px[p_TM][p_TM][p_TM];
        __shared__ float s_d2py[p_TM][p_TM][p_TM];
        __shared__ float s_d2pz[p_TM][p_TM][p_TM];

        /// Load pressure field per thread memory
        cubeThreads {
            const int idxl = i * p_NF + j * p_TM + k * p_TM * p_NF;
            #pragma unroll
            for (int n = 0; n < p_NF; ++n) {
                r_pn[n] = d_pn[idxl + n];
                r_pp[n] = 0.0f;
            }
        }
        __syncthreads();

        /// Calculate second derivatives
        cubeThreads {
            const int idxl = i * p_NF + j * p_TM + k * p_TM * p_NF;
            if (i > 0 && i < p_TM - 1) {
                s_d2px[k][j][i] = (d_pn[idxl + 1] - 2.0f * d_pn[idxl] + d_pn[idxl - 1]) / (dx * dx);
            }
            if (j > 0 && j < p_TM - 1) {
                s_d2py[k][j][i] = (d_pn[idxl + p_TM] - 2.0f * d_pn[idxl] + d_pn[idxl - p_TM]) / (dy * dy);
            }
            if (k > 0 && k < p_TM - 1) {
                s_d2pz[k][j][i] = (d_pn[idxl + p_TM * p_TM] - 2.0f * d_pn[idxl] + d_pn[idxl - p_TM * p_TM]) / (dz * dz);
            }
        }
        __syncthreads();

        /// Compute the wave equation
        cubeThreads {
            const int idxl = i * p_NF + j * p_TM + k * p_TM * p_NF;
            #pragma unroll
            for (int n = 0; n < p_NF; ++n) {
                r_pp[n] = d_v[idx] * d_v[idx] * (s_d2px[k][j][i] + s_d2py[k][j][i] + s_d2pz[k][j][i]) -
                                (r_pn[n] - 2.0f * d_pn[idxl + n]) / (dt * dt);
            }
        }
        __syncthreads();

        /// Update the global residual memory
        cubeThreads {
            const int idxl = i * p_NF + j * p_TM + k * p_TM * p_NF;
            #pragma unroll
            for (int n = 0; n < p_NF; ++n) {
                d_pp[idxl + n] = r_pp[n];
            }
        }
    }
}


/// Using Grid-Stide Loop and OCCA
__global__ void Discontinuous_Galerkin_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    /// Using Grid-Stride Loop defined in macro
    cubeThreads_GridStrideLoop {
        int idx = ix + iy * nx + iz * nx * ny;

        /// Make sure the index is within the domain boundaries
        if (ix >= 1 && ix < nx - 1 && iy >= 1 && iy < ny - 1 && iz >= 1 && iz < nz - 1) {
            float d2px = (d_pn[idx + 1] - 2.0f * d_pn[idx] + d_pn[idx - 1]) / (dx * dx);
            float d2py = (d_pn[idx + nx] - 2.0f * d_pn[idx] + d_pn[idx - nx]) / (dy * dy);
            float d2pz = (d_pn[idx + nx * ny] - 2.0f * d_pn[idx] + d_pn[idx - nx * ny]) / (dz * dz);

            /// Order calculation using Grid-Stride Loop
            d_pp[idx] = d_v[idx] * d_v[idx] * (d2px + d2py + d2pz) - (d_pn[idx] - 2.0f * d_pn[idx]) / (dt * dt);
        }
    }
    __syncthreads();
}


/// Leapfrog menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void Leapfrog_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    float xscale, yscale, zscale;

    xscale = (dt*dt) / (dx*dx);
    yscale = (dt*dt) / (dy*dy);
    zscale = (dt*dt) / (dz*dz);

    if (ix < nx - 12 && iy < ny - 12) {
        int in_idx = (iy + 6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

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
        infront2 = d_pn[in_idx];
        in_idx += stride;
        infront1 = d_pn[in_idx];
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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][threadIdx.x + 6] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][threadIdx.x + 6] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[threadIdx.y + 6][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[threadIdx.y + 6][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.y + 6][threadIdx.x + 6] = current;
            __syncthreads();

            float value = (xscale*dx + yscale*dy + zscale*dz) * current;

            value += (2 * pow(dz, 2) / 2) *
                            zscale * (infront1 + behind1) +
                     (2 * pow(dy, 2) / 2) *
                            yscale * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 7][threadIdx.x + 6]) +
                     (2 * pow(dx, 2) / 2) *
                            xscale * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 7]);

            value += (2 * pow(dz, 4) / 24) *
                            zscale * (infront2 + behind2) +
                     (2 * pow(dy, 4) / 24) *
                            yscale * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 8][threadIdx.x + 6]) +
                     (2 * pow(dx, 4) / 24) *
                            xscale * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 8]);

            value += (2 * pow(dz, 6) / 720) *
                            zscale * (infront3 + behind3) +
                     (2 * pow(dy, 6) / 720) *
                            yscale * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 9][threadIdx.x + 6]) +
                     (2 * pow(dx, 6) / 720) *
                            xscale * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 9]);

            value += (2 * pow(dz, 8) / 40320) *
                            zscale * (infront4 + behind4) +
                     (2 * pow(dy, 8) / 40320) *
                            yscale * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 10][threadIdx.x + 6]) +
                     (2 * pow(dx, 8) / 40320) *
                            xscale * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 10]);

            value += (2 * pow(dz, 10) / 3628800) *
                            zscale * (infront5 + behind5) +
                     (2 * pow(dy, 10) / 3628800) *
                            yscale * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 11][threadIdx.x + 6]) +
                     (2 * pow(dx, 10) / 3628800) *
                            xscale * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 11]);

            value += (2 * pow(dz, 12) / 479001600) *
                            zscale * (infront6 + behind6) +
                     (2 * pow(dy, 12) / 479001600) *
                            yscale * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 12][threadIdx.x + 6]) +
                     (2 * pow(dx, 12) / 479001600) *
                            xscale * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 12]);

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void Leapfrog_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float xscale = (dt*dt) / (dx*dx);
    float yscale = (dt*dt) / (dy*dy);
    float zscale = (dt*dt) / (dz*dz);

    /// Macro to Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;

        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order = 0.0f;
        float factorial = 1.0f;

        float value_laplacian = zscale * (infront + behind - 2.0f * current)
                              + xscale * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + yscale * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int order = 2; order <= 12; order += 2) {
            factorial *= (order - 1) * order;
            float coeff = 1.0f / factorial; /// iterative factorial
            value_high_order += coeff * (zscale * (d_pn[in_idx + order * nx * ny] +
                                                   d_pn[in_idx - order * nx * ny]) +
                                         yscale * (d_pn[in_idx + order * nx] +
                                                   d_pn[in_idx - order * nx]) +
                                         xscale * (d_pn[in_idx + order] +
                                                   d_pn[in_idx - order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order);
    }
    __syncthreads();
}


/// Crank-Nicolson menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void CrankNicolson_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
    float xscale, yscale, zscale;

    xscale = (dt*dt) / (dx*dx) * 0.5;
    yscale = (dt*dt) / (dy*dy) * 0.5;
    zscale = (dt*dt) / (dz*dz) * 0.5;

    if (ix < nx - 12 && iy < ny - 12) {
        int in_idx = (iy + 6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][threadIdx.x + 6] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][threadIdx.x + 6] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[threadIdx.y + 6][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[threadIdx.y + 6][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.y + 6][threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale*dx + yscale*dy + zscale*dz) +
                           (xscale*(dx+dt) + yscale*(dy+dt) + zscale*(dz+dt))) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        (2*zscale) * (infront1 + behind1) +
                     ((2 * pow(dy, 2) / 2) *
                        yscale * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 7][threadIdx.x + 6]) +
                      ((2 * pow(dy, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        yscale * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 7][threadIdx.x + 6])) +
                     ((2 * pow(dx, 2) / 2) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 7]) +
                      ((2 * pow(dx, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        (2*zscale) * (infront2 + behind2) +
                     ((2 * pow(dy, 4) / 24) *
                        yscale * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 8][threadIdx.x + 6]) +
                      ((2 * pow(dy, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        yscale * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 8][threadIdx.x + 6])) +
                     ((2 * pow(dx, 4) / 24) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 8]) +
                      ((2 * pow(dx, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        (2*zscale) * (infront3 + behind3) +
                     ((2 * pow(dy, 6) / 720) *
                        yscale * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 9][threadIdx.x + 6]) +
                      ((2 * pow(dy, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        yscale * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 9][threadIdx.x + 6])) +
                     ((2 * pow(dx, 6) / 720) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 9]) +
                      ((2 * pow(dx, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        (2*zscale) * (infront4 + behind4) +
                     ((2 * pow(dy, 8) / 40320) *
                        yscale * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 10][threadIdx.x + 6]) +
                      ((2 * pow(dy, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        yscale * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 10][threadIdx.x + 6])) +
                     ((2 * pow(dx, 8) / 40320) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 10]) +
                      ((2 * pow(dx, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        (2*zscale) * (infront5 + behind5) +
                     ((2 * pow(dy, 10) / 3628800) *
                        yscale * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 11][threadIdx.x + 6]) +
                      ((2 * pow(dy, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        yscale * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 11][threadIdx.x + 6])) +
                     ((2 * pow(dx, 10) / 3628800) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 11]) +
                      ((2 * pow(dx, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        (2*zscale) * (infront6 + behind6) +
                     ((2 * pow(dy, 12) / 479001600) *
                        yscale * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 12][threadIdx.x + 6]) +
                      ((2 * pow(dy, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        yscale * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 12][threadIdx.x + 6])) +
                     ((2 * pow(dx, 12) / 479001600) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 12]) +
                      ((2 * pow(dx, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void CrankNicolson_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float xscale = 0.5 * ((dt*dt)/(dx*dx));
    float yscale = 0.5 * ((dt*dt)/(dy*dy));
    float zscale = 0.5 * ((dt*dt)/(dz*dz));

    /// Macro to 3D Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;
        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order = 0.0f;
        float factorial = 1.0f;

        float value_laplacian = ((zscale*dz)+(zscale*(dz+dt))) * (infront + behind - 2.0f * current)
                              + ((xscale*dx)+(xscale*(dx+dt))) * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + ((yscale*dy)+(yscale*(dy+dt))) * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int order = 2; order <= 12; order += 2) {
            factorial *= (order - 1) * order;
            float coeff = 1.0f / factorial; /// iterative factorial
            value_high_order += coeff * (((zscale*dz)+(zscale*(dz+dt))) * (d_pn[in_idx + order * nx * ny] +
                                                                           d_pn[in_idx - order * nx * ny]) +
                                         ((yscale*dy)+(yscale*(dy+dt))) * (d_pn[in_idx + order * nx] +
                                                                           d_pn[in_idx - order * nx]) +
                                         ((xscale*dx)+(xscale*(dx+dt))) * (d_pn[in_idx + order] +
                                                                           d_pn[in_idx - order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order);
    }
    __syncthreads();
}


/// ADI menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void ADI_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
    float xscale, yscale, zscale;
    float dt2 = (1/3)*dt;
    float dt3 = (2/3)*dt;

    xscale = 0.5 * ((dt*dt) / (dx*dx));
    yscale = 0.5 * ((dt*dt) / (dy*dy));
    zscale = 0.5 * ((dt*dt) / (dz*dz));

    if (ix < nx-12 && iy < ny-12) {
        int in_idx = (iy+6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][threadIdx.x + 6] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][threadIdx.x + 6] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[threadIdx.y + 6][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[threadIdx.y + 6][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.y + 6][threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale*dx + yscale*dy + zscale*dz) +
                           (xscale*(dx+dt2) + yscale*(dy+dt3) + zscale*(dz+dt))) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        (2*zscale) * (infront1 + behind1) +
                     ((2 * pow(dy, 2) / 2) *
                        yscale * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 7][threadIdx.x + 6]) +
                      ((2 * pow(dy, 2) / 2) + (2 * pow(dt3, 2) / 2)) *
                        yscale * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 7][threadIdx.x + 6])) +
                     ((2 * pow(dx, 2) / 2) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 7]) +
                      ((2 * pow(dx, 2) / 2) + (2 * pow(dt2, 2) / 2)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        (2*zscale) * (infront2 + behind2) +
                     ((2 * pow(dy, 4) / 24) *
                        yscale * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 8][threadIdx.x + 6]) +
                      ((2 * pow(dy, 4) / 24) + (2 * pow(dt3, 4) / 24)) *
                        yscale * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 8][threadIdx.x + 6])) +
                     ((2 * pow(dx, 4) / 24) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 8]) +
                      ((2 * pow(dx, 4) / 24) + (2 * pow(dt2, 4) / 24)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        (2*zscale) * (infront3 + behind3) +
                     ((2 * pow(dy, 6) / 720) *
                        yscale * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 9][threadIdx.x + 6]) +
                      ((2 * pow(dy, 6) / 720) + (2 * pow(dt3, 6) / 720)) *
                        yscale * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 9][threadIdx.x + 6])) +
                     ((2 * pow(dx, 6) / 720) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 9]) +
                      ((2 * pow(dx, 6) / 720) + (2 * pow(dt2, 6) / 720)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        (2*zscale) * (infront4 + behind4) +
                     ((2 * pow(dy, 8) / 40320) *
                        yscale * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 10][threadIdx.x + 6]) +
                      ((2 * pow(dy, 8) / 40320) + (2 * pow(dt3, 8) / 40320)) *
                        yscale * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 10][threadIdx.x + 6])) +
                     ((2 * pow(dx, 8) / 40320) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 10]) +
                      ((2 * pow(dx, 8) / 40320) + (2 * pow(dt2, 8) / 40320)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        (2*zscale) * (infront5 + behind5) +
                     ((2 * pow(dy, 10) / 3628800) *
                        yscale * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 11][threadIdx.x + 6]) +
                      ((2 * pow(dy, 10) / 3628800) + (2 * pow(dt3, 10) / 3628800)) *
                        yscale * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 11][threadIdx.x + 6])) +
                     ((2 * pow(dx, 10) / 3628800) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 11]) +
                      ((2 * pow(dx, 10) / 3628800) + (2 * pow(dt2, 10) / 3628800)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        (2*zscale) * (infront6 + behind6) +
                     ((2 * pow(dy, 12) / 479001600) *
                        yscale * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 12][threadIdx.x + 6]) +
                      ((2 * pow(dy, 12) / 479001600) + (2 * pow(dt3, 12) / 479001600)) *
                        yscale * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                  s_data[threadIdx.y + 12][threadIdx.x + 6])) +
                     ((2 * pow(dx, 12) / 479001600) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 12]) +
                      ((2 * pow(dx, 12) / 479001600) + (2 * pow(dt2, 12) / 479001600)) *
                        xscale * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                  s_data[threadIdx.y + 6][threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void ADI_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float dt2 = (1/3)*dt;
    float dt3 = (2/3)*dt;
    float xscale = 0.5 * ((dt*dt)/(dx*dx));
    float yscale = 0.5 * ((dt*dt)/(dy*dy));
    float zscale = 0.5 * ((dt*dt)/(dz*dz));

    /// Macro to 3D Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;
        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order = 0.0f;
        float factorial = 1.0f;

        float value_laplacian = ((zscale*dz)+(zscale*(dz+dt))) * (infront + behind - 2.0f * current)
                              + ((xscale*dx)+(xscale*(dx+dt2))) * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + ((yscale*dy)+(yscale*(dy+dt3))) * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int order = 2; order <= 12; order += 2) {
            factorial *= (order - 1) * order;
            float coeff = 1.0f / factorial; /// iterative factorial
            value_high_order += coeff * (((zscale*dz)+(zscale*(dz+dt))) * (d_pn[in_idx + order * nx * ny] +
                                                                           d_pn[in_idx - order * nx * ny]) +
                                         ((yscale*dy)+(yscale*(dy+dt2))) * (d_pn[in_idx + order * nx] +
                                                                            d_pn[in_idx - order * nx]) +
                                         ((xscale*dx)+(xscale*(dx+dt3))) * (d_pn[in_idx + order] +
                                                                            d_pn[in_idx - order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order);
    }
    __syncthreads();
}


/// Sigma 1/4 Formulation to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void Sigma_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
    float xscale1, xscale2, yscale1, yscale2, zscale1, zscale2;

    xscale1 = sigma1 * (dt*dt) / (dx*dx);
    xscale2 = sigma2 * (dt*dt) / (dx*dx);

    yscale1 = sigma1 * (dt*dt) / (dy*dy);
    yscale2 = sigma2 * (dt*dt) / (dy*dy);

    zscale1 = sigma1 * (dt*dt) / (dz*dz);
    zscale2 = sigma2 * (dt*dt) / (dz*dz);

    if (ix < nx-12 && iy < ny-12) {
        int in_idx = (iy+6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][threadIdx.x + 6] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][threadIdx.x + 6] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[threadIdx.y + 6][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[threadIdx.y + 6][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.y + 6][threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale1*(dx+dt) + yscale1*(dy+dt) + zscale1*(dz+dt)) +
                           (xscale2*dx + yscale2*dy + zscale2*dz)) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        (zscale1+zscale2) * (infront1 + behind1) +
                     (((2 * pow(dy, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        yscale1 * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 7][threadIdx.x + 6]) +
                      (2 * pow(dy, 2) / 2) *
                        yscale2 * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 7][threadIdx.x + 6])) +
                     (((2 * pow(dx, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        xscale1 * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 7]) +
                      (2 * pow(dx, 2) / 2) *
                        xscale2 * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        (zscale1+zscale2) * (infront2 + behind2) +
                     (((2 * pow(dy, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        yscale1 * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 8][threadIdx.x + 6]) +
                      (2 * pow(dy, 4) / 24) *
                        yscale2 * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 8][threadIdx.x + 6])) +
                     (((2 * pow(dx, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        xscale1 * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 8]) +
                      (2 * pow(dx, 4) / 24) *
                        xscale2 * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        (zscale1+zscale2) * (infront3 + behind3) +
                     (((2 * pow(dy, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        yscale1 * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 9][threadIdx.x + 6]) +
                      (2 * pow(dy, 6) / 720) *
                        yscale2 * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 9][threadIdx.x + 6])) +
                     (((2 * pow(dx, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        xscale1 * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 9]) +
                      (2 * pow(dx, 6) / 720) *
                        xscale2 * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        (zscale1+zscale2) * (infront4 + behind4) +
                     (((2 * pow(dy, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        yscale1 * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 10][threadIdx.x + 6]) +
                      (2 * pow(dy, 8) / 40320) *
                        yscale2 * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 10][threadIdx.x + 6])) +
                     (((2 * pow(dx, 8) / 40320) + (2 * pow(dt, 8) / 40320))*
                        xscale1 * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 10]) +
                       (2 * pow(dx, 8) / 40320) *
                        xscale1 * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        (zscale1+zscale2) * (infront5 + behind5) +
                     (((2 * pow(dy, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        yscale1 * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 11][threadIdx.x + 6]) +
                      (2 * pow(dy, 10) / 3628800) *
                        yscale2 * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 11][threadIdx.x + 6])) +
                     (((2 * pow(dx, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        xscale1 * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 11]) +
                      (2 * pow(dx, 10) / 3628800) *
                        xscale2 * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dz, 12) / 479001600)) *
                        (zscale1+zscale2) * (infront6 + behind6) +
                     (((2 * pow(dy, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        yscale1 * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 12][threadIdx.x + 6]) +
                      (2 * pow(dy, 12) / 479001600) *
                        yscale2 * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                   s_data[threadIdx.y + 12][threadIdx.x + 6])) +
                     (((2 * pow(dx, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        xscale1 * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 12]) +
                      (2 * pow(dx, 12) / 479001600) *
                        xscale2 * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                   s_data[threadIdx.y + 6][threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void Sigma_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float xscale1 = sigma1 * (dt*dt) / (dx*dx);
    float xscale2 = sigma2 * (dt*dt) / (dx*dx);

    float yscale1 = sigma1 * (dt*dt) / (dy*dy);
    float yscale2 = sigma2 * (dt*dt) / (dy*dy);

    float zscale1 = sigma1 * (dt*dt) / (dz*dz);
    float zscale2 = sigma2 * (dt*dt) / (dz*dz);

    /// Macro to 3D Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;
        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order = 0.0f;
        float factorial = 1.0f;

        float value_laplacian = ((zscale2*dz)+(zscale1*(dz+dt))) * (infront + behind - 2.0f * current)
                              + ((xscale2*dx)+(xscale1*(dx+dt))) * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + ((yscale2*dy)+(yscale1*(dy+dt))) * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int order = 2; order <= 12; order += 2) {
            factorial *= (order - 1) * order;
            float coeff = 1.0f / factorial; /// iterative factorial
            value_high_order += coeff * (((zscale2*dz)+(zscale1*(dz+dt))) * (d_pn[in_idx + order * nx * ny] +
                                                                             d_pn[in_idx - order * nx * ny]) +
                                         ((yscale2*dy)+(yscale1*(dy+dt))) * (d_pn[in_idx + order * nx] +
                                                                             d_pn[in_idx - order * nx]) +
                                         ((xscale2*dx)+(xscale1*(dx+dt))) * (d_pn[in_idx + order] +
                                                                             d_pn[in_idx - order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order);
    }
    __syncthreads();
}


/// Lax-Wendroff menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void LaxWendroff_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
    float xscale_courant, xscale_diffusion, yscale_courant, yscale_diffusion;
    float zscale_courant, zscale_diffusion;

    xscale_courant   = 0.5 * dt / dx;
    xscale_diffusion = 0.5 * (dt*dt) / (dx*dx);

    yscale_courant   = 0.5 * dt / dy;
    yscale_diffusion = 0.5 * (dt*dt) / (dy*dy);

    zscale_courant = 0.5 * dt / dz;
    zscale_diffusion = 0.5 * (dt*dt) / (dz*dz);

    if (ix < nx-12 && iy < ny-12) {
        int in_idx = (iy+6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][threadIdx.x + 6] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][threadIdx.x + 6] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[threadIdx.y + 6][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[threadIdx.y + 6][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.y + 6][threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale_diffusion*dx + yscale_diffusion*dy + zscale_diffusion*dz) +
                           (xscale_courant*dx + yscale_courant*dy + zscale_courant*dz)) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dz, 1) / 1)) *
                        (zscale_courant+zscale_diffusion) * (infront1 + behind1) +
                     ((2 * pow(dy, 1) / 1) *
                        yscale_courant * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                          s_data[threadIdx.y + 7][threadIdx.x + 6]) +
                      (2 * pow(dy, 2) / 2) *
                        yscale_diffusion * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 7][threadIdx.x + 6])) +
                     ((2 * pow(dx, 1) / 1) *
                        xscale_courant * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                          s_data[threadIdx.y + 6][threadIdx.x + 7]) +
                      (2 * pow(dx, 2) / 2) *
                        xscale_diffusion * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dz, 3) / 6)) *
                        (zscale_courant+zscale_diffusion) * (infront2 + behind2) +
                     ((2 * pow(dy, 3) / 6) *
                        yscale_courant * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                          s_data[threadIdx.y + 8][threadIdx.x + 6]) +
                      (2 * pow(dy, 4) / 24) *
                        yscale_diffusion * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 8][threadIdx.x + 6])) +
                     ((2 * pow(dx, 3) / 6) *
                        xscale_courant * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                          s_data[threadIdx.y + 6][threadIdx.x + 8]) +
                      (2 * pow(dx, 4) / 24) *
                        xscale_diffusion * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dz, 5) / 120)) *
                        (zscale_courant+zscale_diffusion) * (infront3 + behind3) +
                     ((2 * pow(dy, 5) / 120) *
                        yscale_courant * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                          s_data[threadIdx.y + 9][threadIdx.x + 6]) +
                      (2 * pow(dy, 6) / 720) *
                        yscale_diffusion * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 9][threadIdx.x + 6])) +
                     ((2 * pow(dx, 5) / 120) *
                        xscale_courant * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                          s_data[threadIdx.y + 6][threadIdx.x + 9]) +
                      (2 * pow(dx, 6) / 720) *
                        xscale_diffusion * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dz, 7) / 5040)) *
                        (zscale_courant+zscale_diffusion) * (infront4 + behind4) +
                     ((2 * pow(dy, 7) / 5040) *
                        yscale_courant * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                          s_data[threadIdx.y + 10][threadIdx.x + 6]) +
                      (2 * pow(dy, 8) / 40320) *
                        yscale_diffusion * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 10][threadIdx.x + 6])) +
                     ((2 * pow(dx, 7) / 5040) *
                        xscale_courant * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                          s_data[threadIdx.y + 6][threadIdx.x + 10]) +
                      (2 * pow(dx, 8) / 40320) *
                        xscale_diffusion * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dz, 9) / 362880)) *
                        (zscale_courant+zscale_diffusion) * (infront5 + behind5) +
                     ((2 * pow(dy, 9) / 362880) *
                        yscale_courant * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                          s_data[threadIdx.y + 11][threadIdx.x + 6]) +
                      (2 * pow(dy, 10) / 3628800) *
                        yscale_diffusion * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 11][threadIdx.x + 6])) +
                     ((2 * pow(dx, 9) / 362880) *
                        xscale_courant * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                          s_data[threadIdx.y + 6][threadIdx.x + 11]) +
                      (2 * pow(dx, 10) / 3628800) *
                        xscale_diffusion * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dz, 11) / 39916800)) *
                        (zscale_courant+zscale_diffusion) * (infront6 + behind6) +
                     ((2 * pow(dy, 11) / 39916800) *
                        yscale_courant * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                          s_data[threadIdx.y + 12][threadIdx.x + 6]) +
                      (2 * pow(dy, 12) / 479001600) *
                        yscale_diffusion * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 12][threadIdx.x + 6])) +
                     ((2 * pow(dx, 11) / 39916800) *
                        xscale_courant * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                          s_data[threadIdx.y + 6][threadIdx.x + 12]) +
                      (2 * pow(dx, 12) / 479001600) *
                        xscale_diffusion * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void LaxWendroff_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float xscale_courant   = 0.5 * dt / dx;
    float xscale_diffusion = 0.5 * (dt*dt) / (dx*dx);

    float yscale_courant   = 0.5 * dt / dy;
    float yscale_diffusion = 0.5 * (dt*dt) / (dy*dy);

    float zscale_courant   = 0.5 * dt / dz;
    float zscale_diffusion = 0.5 * (dt*dt) / (dz*dz);

    /// Macro to 3D Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;
        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order = 0.0f;
        float factorial = 1.0f;

        float value_laplacian = ((zscale_courant*dz)+(zscale_diffusion*dz)) * (infront + behind - 2.0f * current)
                              + ((xscale_courant*dx)+(xscale_diffusion*dx)) * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + ((yscale_courant*dy)+(yscale_diffusion*dy)) * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int order = 2; order <= 12; order += 2) {
            factorial *= (order - 1) * order;
            float coeff = 1.0f / factorial; /// iterative factorial
            value_high_order += coeff * (((zscale_courant*dz)+(zscale_diffusion*dz)) * (d_pn[in_idx + order * nx * ny] +
                                                                                        d_pn[in_idx - order * nx * ny]) +
                                         ((yscale_courant*dy)+(yscale_diffusion*dy)) * (d_pn[in_idx + order * nx] +
                                                                                        d_pn[in_idx - order * nx]) +
                                         ((xscale_courant*dx)+(xscale_diffusion*dx)) * (d_pn[in_idx + order] +
                                                                                        d_pn[in_idx - order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order);
    }
    __syncthreads();
}


/// Fractional Step menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void FractionalStep_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
    float xscale, yscale, zscale;
    float dt2 = (1/3)*dt;
    float dt3 = (2/3)*dt;

    xscale = 0.5 * (dt*dt*0.5) / (dx*dx);
    yscale = 0.5 * (dt*dt*0.5) / (dy*dy);
    zscale = 0.5 * (dt*dt*0.5) / (dz*dz);

    if (ix < nx-12 && iy < ny-12) {
        int in_idx = (iy+6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][threadIdx.x + 6] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][threadIdx.x + 6] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[threadIdx.y + 6][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[threadIdx.y + 6][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.y + 6][threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale*(dx+dt2) + yscale*(dy+dt3) + zscale*(dz+dt)) +
                           (xscale*dx + yscale*dy + zscale*dz)) * current;

            value += ((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        (4*zscale) * (infront1 + behind1) +
                     (((2 * pow(dy, 2) / 2) + (2 * pow(dt3, 2) / 2)) *
                        (2*yscale) * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 7][threadIdx.x + 6]) +
                      (2 * pow(dy, 2) / 2) *
                        (2*yscale) * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 7][threadIdx.x + 6])) +
                     (((2 * pow(dx, 2) / 2) + (2 * pow(dt2, 2) / 2)) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 7]) +
                      (2 * pow(dx, 2) / 2) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 7]));

            value += ((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        (4*zscale) * (infront2 + behind2) +
                     (((2 * pow(dy, 4) / 24) + (2 * pow(dt3, 4) / 24)) *
                        (2*yscale) * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 8][threadIdx.x + 6]) +
                      (2 * pow(dy, 4) / 24) *
                        (2*yscale) * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 8][threadIdx.x + 6])) +
                     (((2 * pow(dx, 4) / 24) + (2 * pow(dt2, 4) / 24)) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 8]) +
                      (2 * pow(dx, 4) / 24) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 8]));

            value += ((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        (4*zscale) * (infront3 + behind3) +
                     (((2 * pow(dy, 6) / 720) + (2 * pow(dt3, 6) / 720)) *
                        (2*yscale) * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 9][threadIdx.x + 6]) +
                      (2 * pow(dy, 6) / 720) *
                        (2*yscale) * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 9][threadIdx.x + 6])) +
                     (((2 * pow(dx, 6) / 720) + (2 * pow(dt2, 6) / 720)) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 9]) +
                      (2 * pow(dx, 6) / 720) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 9]));

            value += ((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        (4*zscale) * (infront4 + behind4) +
                     (((2 * pow(dy, 8) / 40320) + (2 * pow(dt3, 8) / 40320)) *
                        (2*yscale) * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 10][threadIdx.x + 6]) +
                      (2 * pow(dy, 8) / 40320) *
                        (2*yscale) * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 10][threadIdx.x + 6])) +
                     (((2 * pow(dx, 8) / 40320) + (2 * pow(dt2, 8) / 40320)) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 10]) +
                      (2 * pow(dx, 8) / 40320) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 10]));

            value += ((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        (4*zscale) * (infront5 + behind5) +
                     (((2 * pow(dy, 10) / 3628800) + (2 * pow(dt3, 10) / 3628800)) *
                        (2*yscale) * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 11][threadIdx.x + 6]) +
                      (2 * pow(dy, 10) / 3628800) *
                        (2*yscale) * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 11][threadIdx.x + 6])) +
                     (((2 * pow(dx, 10) / 3628800) + (2 * pow(dt2, 10) / 3628800)) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 11]) +
                      (2 * pow(dx, 10) / 3628800) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 11]));

            value += ((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        (4*zscale) * (infront6 + behind6) +
                     (((2 * pow(dy, 12) / 479001600) + (2 * pow(dt3, 12) / 479001600)) *
                        (2*yscale) * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 12][threadIdx.x + 6]) +
                      (2 * pow(dy, 12) / 479001600) *
                        (2*yscale) * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 12][threadIdx.x + 6])) +
                     (((2 * pow(dx, 12) / 479001600) + (2 * pow(dt2, 12) / 479001600)) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 12]) +
                      (2 * pow(dx, 12) / 479001600) *
                        (2*xscale) * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void FractionalStep_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float dt2 = (1/3)*dt;
    float dt3 = (2/3)*dt;
    float xscale = 0.5 * ((dt*dt*0.5)/(dx*dx));
    float yscale = 0.5 * ((dt*dt*0.5)/(dy*dy));
    float zscale = 0.5 * ((dt*dt*0.5)/(dz*dz));

    /// Macro to 3D Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;
        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order = 0.0f;
        float factorial = 1.0f;

        float value_laplacian = ((zscale*dz)+(zscale*(dz+dt))) * (infront + behind - 2.0f * current)
                              + ((xscale*dx)+(xscale*(dx+dt2))) * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + ((yscale*dy)+(yscale*(dy+dt3))) * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int order = 2; order <= 12; order += 2) {
            factorial *= (order - 1) * order;
            float coeff = 1.0f / factorial; /// iterative factorial
            value_high_order += coeff * (((2*zscale*dz)+(2*zscale*(dz+dt))) * (d_pn[in_idx + order * nx * ny] +
                                                                               d_pn[in_idx - order * nx * ny]) +
                                         ((2*yscale*dy)+(2*yscale*(dy+dt2))) * (d_pn[in_idx + order * nx] +
                                                                                d_pn[in_idx - order * nx]) +
                                         ((2*xscale*dx)+(2*xscale*(dx+dt3))) * (d_pn[in_idx + order] +
                                                                                d_pn[in_idx - order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order);
    }
    __syncthreads();
}


/// MacCormack menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void MacCormack_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
    float xscale_predictor, xscale_corrector, yscale_predictor, yscale_corrector;
    float zscale_predictor, zscale_corrector;

    xscale_predictor = 0.5 * dt / dx;
    xscale_corrector = (dt*dt) / (dx*dx);

    yscale_predictor = 0.5 * dt / dy;
    yscale_corrector = (dt*dt) / (dy*dy);

    zscale_predictor = 0.5 * dt / dz;
    zscale_corrector = (dt*dt) / (dz*dz);

    if (ix < nx-12 && iy < ny-12) {
        int in_idx = (iy+6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][threadIdx.x + 6] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][threadIdx.x + 6] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[threadIdx.y + 6][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[threadIdx.y + 6][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.y + 6][threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale_corrector*(dx+dt) + yscale_corrector*(dy+dt) + zscale_corrector*(dz+dt)) +
                           (xscale_predictor*dx + yscale_predictor*dy + zscale_predictor*dz)) * current;

            value += (((2 * pow(dz, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        zscale_corrector * (infront1 + behind1) +
                      (2 * pow(dz, 1) / 1) *
                        zscale_predictor * (infront1 + behind1)) +
                     (((2 * pow(dy, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        yscale_corrector * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 7][threadIdx.x + 6]) +
                      (2 * pow(dy, 1) / 1) *
                        yscale_predictor * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 7][threadIdx.x + 6])) +
                     (((2 * pow(dx, 2) / 2) + (2 * pow(dt, 2) / 2)) *
                        xscale_corrector * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 7]) +
                      (2 * pow(dx, 1) / 1) *
                        xscale_predictor * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 7]));

            value += (((2 * pow(dz, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        zscale_corrector * (infront2 + behind2) +
                      (2 * pow(dz, 3) / 6) *
                        zscale_predictor * (infront2 + behind2)) +
                     (((2 * pow(dy, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        yscale_corrector * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 8][threadIdx.x + 6]) +
                      (2 * pow(dy, 3) / 6) *
                        yscale_predictor * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 8][threadIdx.x + 6])) +
                     (((2 * pow(dx, 4) / 24) + (2 * pow(dt, 4) / 24)) *
                        xscale_corrector * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 8]) +
                      (2 * pow(dx, 3) / 6) *
                        xscale_predictor * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 8]));

            value += (((2 * pow(dz, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        zscale_corrector * (infront3 + behind3) +
                      (2 * pow(dz, 5) / 120) *
                        zscale_predictor * (infront3 + behind3)) +
                     (((2 *pow(dy, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        yscale_corrector * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 9][threadIdx.x + 6]) +
                      (2 * pow(dy, 5) / 120) *
                        yscale_predictor * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 9][threadIdx.x + 6])) +
                     (((2 * pow(dx, 6) / 720) + (2 * pow(dt, 6) / 720)) *
                        xscale_corrector * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 9]) +
                      (2 * pow(dx, 5) / 120) *
                        xscale_predictor * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 9]));

            value += (((2 * pow(dz, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        zscale_corrector * (infront4 + behind4) +
                      (2 * pow(dz, 7) / 5040) *
                        zscale_predictor * (infront4 + behind4)) +
                     (((2 * pow(dy, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        yscale_corrector * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 10][threadIdx.x + 6]) +
                      (2 * pow(dy, 7) / 5040) *
                        yscale_predictor * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 10][threadIdx.x + 6])) +
                     (((2 * pow(dx, 8) / 40320) + (2 * pow(dt, 8) / 40320)) *
                        xscale_corrector * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 10]) +
                      (2 * pow(dx, 7) / 5040) *
                        xscale_predictor * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 10]));

            value += (((2 * pow(dz, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        zscale_corrector * (infront5 + behind5) +
                      (2 * pow(dz, 9) / 362880) *
                        zscale_predictor * (infront5 + behind5)) +
                     (((2 * pow(dy, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        yscale_corrector * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 11][threadIdx.x + 6]) +
                      (2 * pow(dy, 9) / 362880) *
                        yscale_predictor * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 11][threadIdx.x + 6])) +
                     (((2 * pow(dx, 10) / 3628800) + (2 * pow(dt, 10) / 3628800)) *
                        xscale_corrector * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 11]) +
                      (2 * pow(dx, 9) / 362880) *
                        xscale_predictor * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 11]));

            value += (((2 * pow(dz, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        zscale_corrector * (infront6 + behind6) +
                      (2 * pow(dz, 11) / 39916800) *
                        zscale_predictor * (infront6 + behind6)) +
                     (((2 * pow(dy, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        yscale_corrector * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 12][threadIdx.x + 6]) +
                      (2 * pow(dy, 11) / 39916800) *
                        yscale_predictor * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                            s_data[threadIdx.y + 12][threadIdx.x + 6])) +
                     (((2 * pow(dx, 12) / 479001600) + (2 * pow(dt, 12) / 479001600)) *
                        xscale_corrector * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 12]) +
                      (2 * pow(dx, 11) / 39916800) *
                        xscale_predictor * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                            s_data[threadIdx.y + 6][threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void MacCormack_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float xscale_predictor = 0.5 * dt / dx;
    float xscale_corrector = (dt*dt) / (dx*dx);

    float yscale_predictor = 0.5 * dt / dy;
    float yscale_corrector = (dt*dt) / (dy*dy);

    float zscale_predictor = 0.5 * dt / dz;
    float zscale_corrector = (dt*dt) / (dz*dz);

    /// Macro to 3D Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;
        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order = 0.0f;
        float factorial = 1.0f;

        float value_laplacian = ((zscale_corrector*(dz*dt))+(zscale_predictor*dz)) * (infront + behind - 2.0f * current)
                              + ((xscale_corrector*(dx*dt))+(xscale_predictor*dx)) * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + ((yscale_corrector*(dy*dt))+(yscale_predictor*dy)) * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int order = 2; order <= 12; order += 2) {
            factorial *= (order - 1) * order;
            float coeff = 1.0f / factorial; /// iterative factorial
            value_high_order += coeff * (((zscale_corrector*(dz*dt))+(zscale_predictor*dz)) * (d_pn[in_idx + order * nx * ny] +
                                                                                               d_pn[in_idx - order * nx * ny]) +
                                         ((yscale_corrector*(dy*dt))+(yscale_predictor*dy)) * (d_pn[in_idx + order * nx] +
                                                                                               d_pn[in_idx - order * nx]) +
                                         ((xscale_corrector*(dx*dt))+(xscale_predictor*dx)) * (d_pn[in_idx + order] +
                                                                                               d_pn[in_idx - order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order);
    }
    __syncthreads();
}


/// Total Variation DIminishing (TVD) menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void TVD_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    float xscale_first, xscale_second, yscale_first, yscale_second;
    float zscale_first, zscale_second;

    xscale_first = 0.5 * dt / dx;
    xscale_second = dt / (dx*dx);

    yscale_first = 0.5 * dt / dy;
    yscale_second = dt / (dy*dy);

    zscale_first = 0.5 * dt / dz;
    zscale_second = dt / (dz*dz);

    if (ix < nx-12 && iy < ny-12) {
        int in_idx = (iy+6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        float coefficient_1 = 2*dx + 2*dy + 2*dz;
        float coefficient_21 = 2*pow(dx,1)/1 + 2*pow(dy,1)/1 + 2*pow(dz,1)/1;
        float coefficient_22 = 2*pow(dx,2)/2 + 2*pow(dy,2)/2 + 2*pow(dz,2)/2;
        float coefficient_31 = 2*pow(dx,3)/6 + 2*pow(dy,3)/6 + 2*pow(dz,3)/6;
        float coefficient_32 = 2*pow(dx,4)/24 + 2*pow(dy,4)/24 + 2*pow(dz,4)/24;
        float coefficient_41 = 2*pow(dx,5)/120 + 2*pow(dy,5)/120 + 2*pow(dz,5)/120;
        float coefficient_42 = 2*pow(dx,6)/720 + 2*pow(dy,6)/720 + 2*pow(dz,6)/720;
        float coefficient_51 = 2*pow(dx,7)/5040 + 2*pow(dy,7)/5040 + 2*pow(dz,7)/5040;
        float coefficient_52 = 2*pow(dx,8)/40320 + 2*pow(dy,8)/40320 + 2*pow(dz,8)/40320;
        float coefficient_61 = 2*pow(dx,9)/362880 + 2*pow(dy,9)/362880 + 2*pow(dz,9)/362880;
        float coefficient_62 = 2*pow(dx,10)/3628800 + 2*pow(dy,10)/3628800 + 2*pow(dz,10)/3628800;
        float coefficient_71 = 2*pow(dx,11)/39916800 + 2*pow(dy,11)/39916800 + 2*pow(dz,11)/39916800;
        float coefficient_72 = 2*pow(dx,12)/479001600 + 2*pow(dy,12)/479001600 + 2*pow(dz,12)/479001600;

        /// define index thread
        int tx = threadIdx.x + 6;
        int ty = threadIdx.y + 6;

        /// Symmetric Flux Limiters from Waterson & Deconinck, 2007.
        /// Paper: "Design principles for bounded higher-order convection schemes – a unified approach"
        int flx1 = (1.5*(pow((tx-1)/(tx+1),2)+((tx-1)/(tx+1)))) / (pow((tx-1)/(tx+1),2)+((tx-1)/(tx+1))+1);
        int flx2 = (1.5*(pow((tx-2)/(tx+2),2)+((tx-2)/(tx+2)))) / (pow((tx-2)/(tx+2),2)+((tx-2)/(tx+2))+1);
        int flx3 = (1.5*(pow((tx-3)/(tx+4),2)+((tx-3)/(tx+3)))) / (pow((tx-3)/(tx+3),2)+((tx-3)/(tx+3))+1);
        int flx4 = (1.5*(pow((tx-4)/(tx+4),2)+((tx-4)/(tx+4)))) / (pow((tx-4)/(tx+4),2)+((tx-4)/(tx+4))+1);
        int flx5 = (1.5*(pow((tx-5)/(tx+5),2)+((tx-5)/(tx+5)))) / (pow((tx-5)/(tx+5),2)+((tx-5)/(tx+5))+1);
        int flx6 = (1.5*(pow((tx-6)/(tx+6),2)+((tx-6)/(tx+6)))) / (pow((tx-6)/(tx+6),2)+((tx-6)/(tx+6))+1);

        int fly1 = (1.5*(pow((ty-1)/(ty+1),2)+((ty-1)/(ty+1)))) / (pow((ty-1)/(ty+1),2)+((ty-1)/(ty+1))+1);
        int fly2 = (1.5*(pow((ty-2)/(ty+2),2)+((ty-2)/(ty+2)))) / (pow((ty-2)/(ty+2),2)+((ty-2)/(ty+2))+1);
        int fly3 = (1.5*(pow((ty-3)/(ty+4),2)+((ty-3)/(ty+3)))) / (pow((ty-3)/(ty+3),2)+((ty-3)/(ty+3))+1);
        int fly4 = (1.5*(pow((ty-4)/(ty+4),2)+((ty-4)/(ty+4)))) / (pow((ty-4)/(ty+4),2)+((ty-4)/(ty+4))+1);
        int fly5 = (1.5*(pow((ty-5)/(ty+5),2)+((ty-5)/(ty+5)))) / (pow((ty-5)/(ty+5),2)+((ty-5)/(ty+5))+1);
        int fly6 = (1.5*(pow((ty-6)/(ty+6),2)+((ty-6)/(ty+6)))) / (pow((ty-6)/(ty+6),2)+((ty-6)/(ty+6))+1);

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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][tx] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][tx] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[ty][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[ty][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[ty][tx] = current;
            __syncthreads();

            float value = (xscale_first+xscale_second + yscale_first+yscale_second +
                           zscale_first+zscale_second) * (coefficient_1) * current;

            value += ((coefficient_21 * (zscale_first * (infront1 + behind1) +
                                         yscale_first * (s_data[ty - 1 - fly1][tx] +
                                                         s_data[ty + 1 + fly1][tx]) +
                                         xscale_first * (s_data[ty][tx - 1 - flx1] +
                                                         s_data[ty][tx + 1 + flx1]))) +
                      (coefficient_22 * (zscale_second * (infront1 + behind1) +
                                         yscale_second * (s_data[ty - 1][tx] +
                                                          s_data[ty + 1][tx]) +
                                         xscale_second * (s_data[ty][tx - 1] +
                                                          s_data[ty][tx + 1]))));

            value += ((coefficient_31 * (zscale_first * (infront2 + behind2) +
                                         yscale_first * (s_data[ty - 2 - fly2][tx] +
                                                         s_data[ty + 2 + fly2][tx]) +
                                         xscale_first * (s_data[ty][tx - 2 - flx2] +
                                                         s_data[ty][tx + 2 + flx2]))) +
                      (coefficient_32 * (zscale_second * (infront2 + behind2) +
                                         yscale_second * (s_data[ty - 2][tx] +
                                                          s_data[ty + 2][tx]) +
                                         xscale_second * (s_data[ty][tx - 2] +
                                                          s_data[ty][tx + 2]))));

            value += ((coefficient_41 * (zscale_first * (infront3 + behind3) +
                                         yscale_first * (s_data[ty - 3 - fly3][tx] +
                                                         s_data[ty + 3 + fly3][tx]) +
                                         xscale_first * (s_data[ty][tx - 3 - flx3] +
                                                         s_data[ty][tx + 3 + flx3]))) +
                      (coefficient_42 * (zscale_second * (infront4 + behind4) +
                                         yscale_second * (s_data[ty - 3][tx] +
                                                          s_data[ty + 3][tx]) +
                                         xscale_second * (s_data[ty][tx - 3] +
                                                          s_data[ty][tx + 3]))));

            value += ((coefficient_51 * (zscale_first * (infront4 + behind4) +
                                         yscale_first * (s_data[ty - 4 - fly4][tx] +
                                                         s_data[ty + 4 + fly4][tx]) +
                                         xscale_first * (s_data[ty][tx - 4 - flx4] +
                                                         s_data[ty][tx + 4 + flx4]))) +
                      (coefficient_52 * (zscale_second * (infront5 + behind5) +
                                         yscale_second * (s_data[ty - 4][tx] +
                                                          s_data[ty + 4][tx]) +
                                         xscale_second * (s_data[ty][tx - 4] +
                                                          s_data[ty][tx + 4]))));

            value += ((coefficient_61 * (zscale_first * (infront5 + behind5) +
                                         yscale_first * (s_data[ty - 5 - fly5][tx] +
                                                         s_data[ty + 5 + fly5][tx]) +
                                         xscale_first * (s_data[ty][tx - 5 - flx5] +
                                                         s_data[ty][tx + 5 + flx5]))) +
                      (coefficient_62 * (zscale_second * (infront5 + behind5) +
                                         yscale_second * (s_data[ty - 5][tx] +
                                                          s_data[ty + 5][tx]) +
                                         xscale_second * (s_data[ty][tx - 5] +
                                                          s_data[ty][tx + 5]))));

            value += ((coefficient_71 * (zscale_first * (infront6 + behind6) +
                                         yscale_first * (s_data[ty - 6 - fly6][tx] +
                                                         s_data[ty + 6 + fly6][tx]) +
                                         xscale_first * (s_data[ty][tx - 6 - flx6] +
                                                         s_data[ty][tx + 6 + flx6]))) +
                      (coefficient_72 * (zscale_second * (infront6 + behind6) +
                                         yscale_second * (s_data[ty - 6][tx] +
                                                          s_data[ty + 6][tx]) +
                                         xscale_second * (s_data[ty][tx - 6] +
                                                          s_data[ty][tx + 6]))));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void TVD_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float xscale_first = 0.5 * dt / dx;
    float xscale_second = (dt*dt) / (dx*dx);

    float yscale_first = 0.5 * dt / dy;
    float yscale_second = (dt*dt) / (dy*dy);

    float zscale_first = 0.5 * dt / dz;
    float zscale_second = (dt*dt) / (dz*dz);

    /// Macro to 3D Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;
        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order_even = 0.0f;
        float value_high_order_odd = 0.0f;
        float factorial = 1.0f;

        /// define index thread
        int tx = threadIdx.x + 6;
        int ty = threadIdx.y + 6;

        /// Symmetric Flux Limiters from Waterson & Deconinck, 2007.
        /// Paper: "Design principles for bounded higher-order convection schemes – a unified approach"
        int flx1 = (1.5*(pow((tx-1)/(tx+1),2)+((tx-1)/(tx+1)))) / (pow((tx-1)/(tx+1),2)+((tx-1)/(tx+1))+1);
        int flx2 = (1.5*(pow((tx-2)/(tx+2),2)+((tx-2)/(tx+2)))) / (pow((tx-2)/(tx+2),2)+((tx-2)/(tx+2))+1);
        int flx3 = (1.5*(pow((tx-3)/(tx+4),2)+((tx-3)/(tx+3)))) / (pow((tx-3)/(tx+3),2)+((tx-3)/(tx+3))+1);
        int flx4 = (1.5*(pow((tx-4)/(tx+4),2)+((tx-4)/(tx+4)))) / (pow((tx-4)/(tx+4),2)+((tx-4)/(tx+4))+1);
        int flx5 = (1.5*(pow((tx-5)/(tx+5),2)+((tx-5)/(tx+5)))) / (pow((tx-5)/(tx+5),2)+((tx-5)/(tx+5))+1);
        int flx6 = (1.5*(pow((tx-6)/(tx+6),2)+((tx-6)/(tx+6)))) / (pow((tx-6)/(tx+6),2)+((tx-6)/(tx+6))+1);

        int fly1 = (1.5*(pow((ty-1)/(ty+1),2)+((ty-1)/(ty+1)))) / (pow((ty-1)/(ty+1),2)+((ty-1)/(ty+1))+1);
        int fly2 = (1.5*(pow((ty-2)/(ty+2),2)+((ty-2)/(ty+2)))) / (pow((ty-2)/(ty+2),2)+((ty-2)/(ty+2))+1);
        int fly3 = (1.5*(pow((ty-3)/(ty+4),2)+((ty-3)/(ty+3)))) / (pow((ty-3)/(ty+3),2)+((ty-3)/(ty+3))+1);
        int fly4 = (1.5*(pow((ty-4)/(ty+4),2)+((ty-4)/(ty+4)))) / (pow((ty-4)/(ty+4),2)+((ty-4)/(ty+4))+1);
        int fly5 = (1.5*(pow((ty-5)/(ty+5),2)+((ty-5)/(ty+5)))) / (pow((ty-5)/(ty+5),2)+((ty-5)/(ty+5))+1);
        int fly6 = (1.5*(pow((ty-6)/(ty+6),2)+((ty-6)/(ty+6)))) / (pow((ty-6)/(ty+6),2)+((ty-6)/(ty+6))+1);

        /*
            value += ((coefficient_21 * (zscale_first * (infront1 + behind1) +
                                         yscale_first * (s_data[ty - 1 - fly1][tx] +
                                                         s_data[ty + 1 + fly1][tx]) +
                                         xscale_first * (s_data[ty][tx - 1 - flx1] +
                                                         s_data[ty][tx + 1 + flx1]))) +
                      (coefficient_22 * (zscale_second * (infront1 + behind1) +
                                         yscale_second * (s_data[ty - 1][tx] +
                                                          s_data[ty + 1][tx]) +
                                         xscale_second * (s_data[ty][tx - 1] +
                                                          s_data[ty][tx + 1]))));
        */

        float value_laplacian = (zscale_first+zscale_second) * (infront + behind - 2.0f * current)
                              + (xscale_first+xscale_second) * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + (yscale_first+yscale_second) * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int even_order = 2; even_order <= 12; even_order += 2) {
            factorial *= (even_order - 1) * even_order;
            float coeff_even = 1.0f / factorial; /// iterative factorial for even numbers
            value_high_order_even += coeff_even * (zscale_second * (d_pn[in_idx + even_order * nx * ny] +
                                                                    d_pn[in_idx - even_order * nx * ny]) +
                                                   yscale_second * (d_pn[in_idx + even_order * nx] +
                                                                    d_pn[in_idx - even_order * nx]) +
                                                   zscale_second * (d_pn[in_idx + even_order] +
                                                                    d_pn[in_idx - even_order]));
        }

        #pragma unroll
        for (int odd_order = 1; odd_order <= 11; odd_order +=2) {
            factorial *= odd_order;
            float coeff_odd = 1.0f / factorial; /// iterative factorial for odd numbers
            value_high_order_odd += coeff_odd * (zscale_first * (d_pn[in_idx + odd_order * nx * ny] +
                                                                 d_pn[in_idx - odd_order * nx * ny]) +
                                                 yscale_first * (d_pn[in_idx + odd_order * nx] +
                                                                 d_pn[in_idx - odd_order * nx]) +
                                                 zscale_first * (d_pn[in_idx + odd_order] +
                                                                 d_pn[in_idx - odd_order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order_even+value_high_order_odd);
    }
    __syncthreads();
}


/// PSOR menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void PSOR_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    float xscale, yscale, zscale;
    float first_scale, second_scale, third_scale, sor1, sor2, a1, a2, wopt1, wopt2;

    xscale = (dt*dt) / (dx*dx);
    yscale = (dt*dt) / (dy*dy);
    zscale = (dt*dt) / (dz*dz);

    /// Based on Hoffmann(2000) approximation
    sor1 = xscale / zscale;
    sor2 = yscale / zscale;
    a1 = powf((cos(M_PI/(IM-1)) + (sor1*sor1)*cos(M_PI/(JM-1))) / (1+sor1*sor1), 2);
    a2 = powf((cos(M_PI/(IM-1)) + (sor2*sor2)*cos(M_PI/(JM-1))) / (1+sor2*sor2), 2);
    wopt1 = (2-2*sqrt(1-a1))/a1;
    wopt2 = (2-2*sqrt(1-a2))/a2;

    first_scale = ((wopt1+wopt2)/2) / (2*(1+sor1*sor2));
    second_scale = wopt1*(sor1*sor1) / (2*(1+sor1*sor1));
    third_scale = wopt2*(sor2*sor2) / (2*(1+sor2*sor2));

    if (ix < nx-12 && iy < ny-12) {
        int in_idx = (iy+6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

        float infront1, infront2, infront3, infront4, infront5, infront6;
        float behind1, behind2, behind3, behind4, behind5, behind6;
        float current;

        float coefficient_1 = dx+dy+dz;
        float coefficient_2 = 2*pow(dx,2)/2 + 2*pow(dy,2)/2 + 2*pow(dz,2)/2;
        float coefficient_3 = 2*pow(dx,4)/24 + 2*pow(dy,4)/24 + 2*pow(dz,4)/24;
        float coefficient_4 = 2*pow(dx,6)/720 + 2*pow(dy,6)/720 + 2*pow(dz,6)/720;
        float coefficient_5 = 2*pow(dx,8)/40320 + 2*pow(dy,8)/40320 + 2*pow(dz,8)/40320;
        float coefficient_6 = 2*pow(dx,10)/3628800 + 2*pow(dy,10)/3628800 + 2*pow(dz,10)/3628800;
        float coefficient_7 = 2*pow(dx,12)/479001600 + 2*pow(dy,12)/479001600 + 2*pow(dz,12)/479001600;

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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][threadIdx.x + 6] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][threadIdx.x + 6] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[threadIdx.y + 6][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[threadIdx.y + 6][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.y + 6][threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((first_scale+second_scale+third_scale)*(wopt1+wopt2)) * (coefficient_1) * current;

            value += coefficient_2 * (first_scale * (infront1 + behind1) +
                                      second_scale * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                                      s_data[threadIdx.y + 7][threadIdx.x + 6]) +
                                      third_scale * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                                     s_data[threadIdx.y + 6][threadIdx.x + 7]));

            value += coefficient_3 * (first_scale * (infront2 + behind2) +
                                      second_scale * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                                      s_data[threadIdx.y + 8][threadIdx.x + 6]) +
                                      third_scale * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                                     s_data[threadIdx.y + 6][threadIdx.x + 8]));

            value += coefficient_4 * (first_scale * (infront3 + behind3) +
                                      second_scale * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                                      s_data[threadIdx.y + 9][threadIdx.x + 6]) +
                                      third_scale * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                                     s_data[threadIdx.y + 6][threadIdx.x + 9]));

            value += coefficient_5 * (first_scale * (infront4 + behind4) +
                                      second_scale * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                                      s_data[threadIdx.y + 10][threadIdx.x + 6]) +
                                      third_scale * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                                     s_data[threadIdx.y + 6][threadIdx.x + 10]));

            value += coefficient_6 * (first_scale * (infront5 + behind5) +
                                      second_scale * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                                      s_data[threadIdx.y + 11][threadIdx.x + 6]) +
                                      third_scale * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                                     s_data[threadIdx.y + 6][threadIdx.x + 11]));

            value += coefficient_7 * (first_scale * (infront6 + behind6) +
                                      second_scale * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                                      s_data[threadIdx.y + 12][threadIdx.x + 6]) +
                                      third_scale * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                                     s_data[threadIdx.y + 6][threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void PSOR_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float xscale = (dt*dt) / (dx*dx);
    float yscale = (dt*dt) / (dy*dy);
    float zscale = (dt*dt) / (dz*dz);

    /// Based on Hoffmann(2000) approximation
    float sor1 = xscale / zscale;
    float sor2 = yscale / zscale;
    float a1 = powf((cos(M_PI/(IM-1)) + (sor1*sor1)*cos(M_PI/(JM-1))) / (1+sor1*sor1), 2);
    float a2 = powf((cos(M_PI/(IM-1)) + (sor2*sor2)*cos(M_PI/(JM-1))) / (1+sor2*sor2), 2);
    float wopt1 = (2-2*sqrt(1-a1))/a1;
    float wopt2 = (2-2*sqrt(1-a2))/a2;

    float first_scale = ((wopt1+wopt2)/2) / (2*(1+sor1*sor2));
    float second_scale = wopt1*(sor1*sor1) / (2*(1+sor1*sor1));
    float third_scale = wopt2*(sor2*sor2) / (2*(1+sor2*sor2));

    /// Macro to Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;

        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order = 0.0f;
        float factorial = 1.0f;

        float value_laplacian = (first_scale*(wopt1+wopt2)) * (infront + behind - 2.0f * current)
                              + (second_scale*(wopt1+wopt2)) * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + (third_scale*(wopt1+wopt2)) * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int order = 2; order <= 12; order += 2) {
            factorial *= (order - 1) * order;
            float coeff = 1.0f / factorial; /// iterative factorial
            value_high_order += coeff * (first_scale * (d_pn[in_idx + order * nx * ny] +
                                                        d_pn[in_idx - order * nx * ny]) +
                                         second_scale * (d_pn[in_idx + order * nx] +
                                                         d_pn[in_idx - order * nx]) +
                                         third_scale * (d_pn[in_idx + order] +
                                                        d_pn[in_idx - order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order);
    }
    __syncthreads();
}


/// Flux-Vector Splitting (FVS) menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void FVS_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    __shared__ float s_data[BLOCK_DIMY + 12][BLOCK_DIMX + 12];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    float xscale_one, xscale_two, yscale_one, yscale_two, zscale_one, zscale_two;

    xscale_one = 0.5 * dt / dx;
    xscale_two = 0.5 * (dt*dt) / (dx*dx);

    yscale_one = 0.5 * dt / dy;
    yscale_two = 0.5 * (dt*dt) / (dy*dy);

    zscale_one = 0.5 * dt / dz;
    zscale_two = 0.5 * (dt*dt) / (dz*dz);

    if (ix < nx-12 && iy < ny-12) {
        int in_idx = (iy+6) * nx + ix + 6;
        int out_idx = 0;
        int stride = nx * ny;

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

            if (threadIdx.y < 6) {
                s_data[threadIdx.y][threadIdx.x + 6] =
                    d_pn[out_idx - 6 * nx];
                s_data[threadIdx.y + BLOCK_DIMY + 6][threadIdx.x + 6] =
                    d_pn[out_idx + BLOCK_DIMY * nx];
            }

            if (threadIdx.x < 6) {
                s_data[threadIdx.y + 6][threadIdx.x] =
                    d_pn[out_idx - 6];
                s_data[threadIdx.y + 6][threadIdx.x + BLOCK_DIMX + 6] =
                    d_pn[out_idx + BLOCK_DIMX];
            }

            s_data[threadIdx.y + 6][threadIdx.x + 6] = current;
            __syncthreads();

            float value = ((xscale_one*dx + yscale_one*dy + zscale_one*dz) +
                           (xscale_two*dx + yscale_two*dy + zscale_two*dz)) * current;

            value += ((2 * pow(dz, 1) / 1) *
                        zscale_one * (infront1 - behind1) +
                      (2 * pow(dz, 2) / 2) *
                        zscale_two * (infront1 + behind1)) +
                     ((2 * pow(dy, 1) / 1) *
                        yscale_one * (s_data[threadIdx.y + 5][threadIdx.x + 6] -
                                      s_data[threadIdx.y + 7][threadIdx.x + 6]) +
                      (2 * pow(dy, 2) / 2) *
                        yscale_two * (s_data[threadIdx.y + 5][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 7][threadIdx.x + 6])) +
                     ((2 * pow(dx, 1) / 1) *
                        xscale_one * (s_data[threadIdx.y + 6][threadIdx.x + 5] -
                                      s_data[threadIdx.y + 6][threadIdx.x + 7]) +
                      (2 *pow(dx, 2) / 2) *
                        xscale_two * (s_data[threadIdx.y + 6][threadIdx.x + 5] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 7]));

            value += ((2 * pow(dz, 3) / 6) *
                        zscale_one * (infront2 - behind2) +
                      (2 * pow(dz, 4) / 24) *
                        zscale_two * (infront2 + behind2)) +
                     ((2 * pow(dy, 3) / 6) *
                        yscale_one * (s_data[threadIdx.y + 4][threadIdx.x + 6] -
                                      s_data[threadIdx.y + 8][threadIdx.x + 6]) +
                      (2 * pow(dy, 4) / 24) *
                        yscale_two * (s_data[threadIdx.y + 4][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 8][threadIdx.x + 6])) +
                     ((2 * pow(dx, 3) / 6) *
                        xscale_one * (s_data[threadIdx.y + 6][threadIdx.x + 4] -
                                      s_data[threadIdx.y + 6][threadIdx.x + 8]) +
                      (2 * pow(dx, 4) / 24) *
                        xscale_two * (s_data[threadIdx.y + 6][threadIdx.x + 4] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 8]));

            value += ((2 * pow(dz, 5) / 120) *
                        zscale_one * (infront3 - behind3) +
                      (2 * pow(dz, 6) / 720) *
                        zscale_two * (infront3 + behind3)) +
                     ((2 * pow(dy, 5) / 120) *
                        yscale_one * (s_data[threadIdx.y + 3][threadIdx.x + 6] -
                                      s_data[threadIdx.y + 9][threadIdx.x + 6]) +
                      (2 * pow(dy, 6) / 720) *
                        yscale_two * (s_data[threadIdx.y + 3][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 9][threadIdx.x + 6])) +
                     ((2 * pow(dx, 5) / 120) *
                        xscale_one * (s_data[threadIdx.y + 6][threadIdx.x + 3] -
                                      s_data[threadIdx.y + 6][threadIdx.x + 9]) +
                      (2 * pow(dx, 6) / 720) *
                        xscale_two * (s_data[threadIdx.y + 6][threadIdx.x + 3] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 9]));

            value += ((2 * pow(dz, 7) / 5040) *
                        zscale_one * (infront4 - behind4) +
                      (2 * pow(dz, 8) / 40320) *
                        zscale_two * (infront4 + behind4)) +
                     ((2 * pow(dy, 7) / 5040) *
                        yscale_one * (s_data[threadIdx.y + 2][threadIdx.x + 6] -
                                      s_data[threadIdx.y + 10][threadIdx.x + 6]) +
                      (2 * pow(dx, 8) / 40320) *
                        yscale_two * (s_data[threadIdx.y + 2][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 10][threadIdx.x + 6])) +
                     ((2 * pow(dx, 7) / 5040) *
                        xscale_one * (s_data[threadIdx.y + 6][threadIdx.x + 2] -
                                      s_data[threadIdx.y + 6][threadIdx.x + 10]) +
                      (2 * pow(dx, 8) / 40320) *
                        xscale_two * (s_data[threadIdx.y + 6][threadIdx.x + 2] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 10]));

            value += ((2 * pow(dz, 9) / 362880) *
                        zscale_one * (infront5 - behind5) +
                      (2 * pow(dz, 10) / 3628800) *
                        zscale_two * (infront5 + behind5)) +
                     ((2 * pow(dy, 9) / 362880) *
                        yscale_one * (s_data[threadIdx.y + 1][threadIdx.x + 6] -
                                      s_data[threadIdx.y + 11][threadIdx.x + 6]) +
                      (2 * pow(dy, 10) / 3628800) *
                        yscale_two * (s_data[threadIdx.y + 1][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 11][threadIdx.x + 6])) +
                     ((2 * pow(dx, 9) / 362880) *
                        xscale_one * (s_data[threadIdx.y + 6][threadIdx.x + 1] -
                                      s_data[threadIdx.y + 6][threadIdx.x + 11]) +
                      (2 * pow(dx, 10) / 3628800) *
                        xscale_two * (s_data[threadIdx.y + 6][threadIdx.x + 1] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 11]));

            value += ((2 * pow(dz, 11) / 39916800) *
                        zscale_one * (infront6 - behind6) +
                      (2 * pow(dz, 12) / 479001600) *
                        zscale_two * (infront6 + behind6)) +
                     ((2 * pow(dy, 11) / 39916800) *
                        yscale_one * (s_data[threadIdx.y + 0][threadIdx.x + 6] -
                                      s_data[threadIdx.y + 12][threadIdx.x + 6]) +
                      (2 * pow(dy, 12) / 479001600) *
                        yscale_two * (s_data[threadIdx.y + 0][threadIdx.x + 6] +
                                      s_data[threadIdx.y + 12][threadIdx.x + 6])) +
                     ((2 * pow(dx, 11) / 39916800) *
                        xscale_one * (s_data[threadIdx.y + 6][threadIdx.x + 0] -
                                      s_data[threadIdx.y + 6][threadIdx.x + 12]) +
                      (2 * pow(dx, 12) / 479001600) *
                        xscale_two * (s_data[threadIdx.y + 6][threadIdx.x + 0] +
                                      s_data[threadIdx.y + 6][threadIdx.x + 12]));

            d_pp[out_idx] = 2.0f * current - d_pp[out_idx] + d_v[out_idx] * value;
        }
    }
}


/// Using Grid-Stride Loop, MultiGPU, and OCCA
__global__ void FVS_3D_Solver_Optimized(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    float xscale_one = 0.5 * dt / dx;
    float xscale_two = 0.5 * (dt*dt) / (dx*dx);

    float yscale_one = 0.5 * dt / dy;
    float yscale_two = 0.5 * (dt*dt) / (dy*dy);

    float zscale_one = 0.5 * dt / dz;
    float zscale_two = 0.5 * (dt*dt) / (dz*dz);

    /// Macro to 3D Grid-Stride Loop and OCCA
#pragma unroll
    cubeThreads_GridStrideLoop {
        int in_idx = (iy+6) * nx + ix + 6 + (iz+6) * nx * ny;
        float current = d_pn[in_idx];
        float infront = d_pn[in_idx + nx * ny];
        float behind = d_pn[in_idx - nx * ny];
        float value_high_order = 0.0f;
        float factorial = 1.0f;

        float value_laplacian = ((zscale_one*dz)+(zscale_two*dz)) * (infront + behind - 2.0f * current)
                              + ((xscale_one*dx)+(xscale_two*dx)) * (d_pn[in_idx - 1] + d_pn[in_idx + 1] - 2.0f * current)
                              + ((yscale_one*dy)+(yscale_two*dy)) * (d_pn[in_idx - nx] + d_pn[in_idx + nx] - 2.0f * current);

        #pragma unroll
        for (int order = 2; order <= 12; order += 2) {
            factorial *= (order - 1) * order;
            float coeff = 1.0f / factorial; /// iterative factorial
            value_high_order += coeff * (((zscale_one*dz)+(zscale_two*dz)) * (d_pn[in_idx + order * nx * ny] +
                                                                              d_pn[in_idx - order * nx * ny]) +
                                         ((yscale_one*dy)+(yscale_two*dy)) * (d_pn[in_idx + order * nx] +
                                                                              d_pn[in_idx - order * nx]) +
                                         ((xscale_one*dx)+(xscale_two*dx)) * (d_pn[in_idx + order] +
                                                                              d_pn[in_idx - order]));
        }

        d_pp[in_idx] = 2.0f * current - d_pp[in_idx] + d_v[in_idx] * (value_laplacian+value_high_order);
    }
    __syncthreads();
}



void Measure_And_Execute_Kernels(
    int nx, int ny, int nz, float dx, float dy, float dz, float dt, float *d_v,
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
        {d_pn_Galerkin, d_pp_Galerkin, h_pp_Galerkin, h_pn_Galerkin, "3D Galerkin Kernel", "GalerkinTime_3D_data.txt", "GalerkinResults_3D_data.bin"},
        {d_pn_Galerkin_Optimized, d_pp_Galerkin_Optimized, h_pp_Galerkin_Optimized, h_pn_Galerkin_Optimized, "3D Galerkin Optimized Kernel", "GalerkinOptimizedTime_3D_data.txt", "GalerkinOptimizedResults_3D_data.bin"},
        {d_pn_Leapfrog, d_pp_Leapfrog, h_pp_Leapfrog, h_pn_Leapfrog, "3D Leapfrog Kernel", "LeapfrogTime_3D_data.txt", "LeapfrogResults_3D_data.bin"},
        {d_pn_Leapfrog_Optimized, d_pp_Leapfrog_Optimized, h_pp_Leapfrog_Optimized, h_pn_Leapfrog_Optimized, "3D Leapfrog Optimized Kernel", "LeapfrogOptimizedTime_3D_data.txt", "LeapfrogOptimizedResults_3D_data.bin"},
        {d_pn_CrankNicolson, d_pp_CrankNicolson, h_pp_CrankNicolson, h_pn_CrankNicolson, "3D Crank-Nicolson Kernel", "CrankNicolsonTime_3D_data.txt", "CrankNicolsonResults_3D_data.bin"},
        {d_pn_CrankNicolson_Optimized, d_pp_CrankNicolson_Optimized, h_pp_CrankNicolson_Optimized, h_pn_CrankNicolson_Optimized, "3D Crank-Nicolson Optimized Kernel", "CrankNicolsonOptimizedTime_3D_data.txt", "CrankNicolsonOptimizedResults_3D_data.bin"},
        {d_pn_ADI, d_pp_ADI, h_pp_ADI, h_pn_ADI, "3D ADI Kernel", "ADITime_3D_data.txt", "ADIResults_3D_data.bin"},
        {d_pn_ADI_Optimized, d_pp_ADI_Optimized, h_pp_ADI_Optimized, h_pn_ADI_Optimized, "3D ADI Optimized Kernel", "ADIOptimizedTime_3D_data.txt", "ADIOptimizedResults_3D_data.bin"},
        {d_pn_Sigma, d_pp_Sigma, h_pp_Sigma, h_pn_Sigma, "3D Sigma Kernel", "SigmaTime_3D_data.txt", "SigmaResults_3D_data.bin"},
        {d_pn_Sigma_Optimized, d_pp_Sigma_Optimized, h_pp_Sigma_Optimized, h_pn_Sigma_Optimized, "3D Sigma Optimized Kernel", "SigmaOptimizedTime_3D_data.txt", "SigmaOptimizedResults_3D_data.bin"},
        {d_pn_LaxWendroff, d_pp_LaxWendroff, h_pp_LaxWendroff, h_pn_LaxWendroff, "3D Lax-Wendroff Kernel", "LaxWendroffTime_3D_data.txt", "LaxWendroffResults_3D_data.bin"},
        {d_pn_LaxWendroff_Optimized, d_pp_LaxWendroff_Optimized, h_pp_LaxWendroff_Optimized, h_pn_LaxWendroff_Optimized, "3D Lax-Wendroff Optimized Kernel", "LaxWendroffOptimizedTime_3D_data.txt", "LaxWendroffOptimizedResults_3D_data.bin"},
        {d_pn_FractionalStep, d_pp_FractionalStep, h_pp_FractionalStep, h_pn_FractionalStep, "3D Fractional Step Kernel", "FractionalStepTime_3D_data.txt", "FractionalStepResults_3D_data.bin"},
        {d_pn_FractionalStep_Optimized, d_pp_FractionalStep_Optimized, h_pp_FractionalStep_Optimized, h_pn_FractionalStep_Optimized, "3D Fractional Step Optimized Kernel", "FractionalStepOptimizedTime_3D_data.txt", "FractionalStepOptimizedResults_3D_data.bin"},
        {d_pn_MacCormack, d_pp_MacCormack, h_pp_MacCormack, h_pn_MacCormack, "3D MacCormack Kernel", "MacCormackTime_3D_data.txt", "MacCormackResults_3D_data.bin"},
        {d_pn_MacCormack_Optimized, d_pp_MacCormack_Optimized, h_pp_MacCormack_Optimized, h_pn_MacCormack_Optimized, "3D MacCormack Optimized Kernel", "MacCormackOptimizedTime_3D_data.txt", "MacCormackOptimizedResults_3D_data.bin"},
        {d_pn_TVD, d_pp_TVD, h_pp_TVD, h_pn_TVD, "3D TVD Kernel", "TVDTime_3D_data.txt", "TVDResults_3D_data.bin"},
        {d_pn_TVD_Optimized, d_pp_TVD_Optimized, h_pp_TVD_Optimized, h_pn_TVD_Optimized, "3D TVD Optimized Kernel", "TVDOptimizedTime_3D_data.txt", "TVDOptimizedResults_3D_data.bin"},
        {d_pn_PSOR, d_pp_PSOR, h_pp_PSOR, h_pn_PSOR, "3D PSOR Kernel", "PSORTime_3D_data.txt", "PSORResults_3D_data.bin"},
        {d_pn_PSOR_Optimized, d_pp_PSOR_Optimized, h_pp_PSOR_Optimized, h_pn_PSOR_Optimized, "3D PSOR Optimized Kernel", "PSOROptimizedTime_3D_data.txt", "PSOROptimizedResults_3D_data.bin"},
        {d_pn_FVS, d_pp_FVS, h_pp_FVS, h_pn_FVS, "3D FVS Kernel", "FVSTime_3D_data.txt", "FVSResults_3D_data.bin"},
        {d_pn_FVS_Optimized, d_pp_FVS_Optimized, h_pp_FVS_Optimized, h_pn_FVS_Optimized, "3D FVS Optimized Kernel", "FVSOptimizedTime_3D_data.txt", "FVSOptimizedResults_3D_data.bin"},
    };

    /// Number of kernel based on kernelData[]
    int number_of_kernels = sizeof(kernelData) / sizeof(kernelData[0]);

    /// Array to save execution time data
    float executionTimes[number_of_kernels][MAX_GPU][NUM_EXECUTIONS] = {0};

#pragma unroll
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        int chunk_size = (nz / num_gpus) + (gpu < remainder ? 1 : 0);
        int z_offset = (gpu < remainder) ? gpu * chunk_size : gpu * (nz / num_gpus) + remainder;
        float *d_v_chunk = d_v + z_offset * nx * ny;

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

                dim3 blockDim(8, 8, 8);
                dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                            (ny + blockDim.y - 1) / blockDim.y,
                            (chunk_size + blockDim.z - 1) / blockDim.z);

                dim3 blockSize1(8, 8, 8);
                dim3 gridSize1((nx + blockSize1.x - 1) / blockSize1.x);

                dim3 blockSize2(8, 8);
                dim3 gridSize2((nx + blockSize2.x - 1) / blockSize2.x);

                switch (i) {
                    case 0:
                        Discontinuous_Galerkin_3D_Solver<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 1:
                        Discontinuous_Galerkin_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 2:
                        Leapfrog_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 3:
                        Leapfrog_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 4:
                        CrankNicolson_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 5:
                        CrankNicolson_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 6:
                        ADI_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 7:
                        ADI_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 8:
                        Sigma_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 9:
                        Sigma_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 10:
                        LaxWendroff_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 11:
                        LaxWendroff_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 12:
                        FractionalStep_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 13:
                        FractionalStep_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 14:
                        MacCormack_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 15:
                        MacCormack_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 16:
                        TVD_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 17:
                        TVD_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 18:
                        PSOR_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 19:
                        PSOR_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 20:
                        FVS_3D_Solver<<<gridSize2, blockSize2, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                    case 21:
                        FVS_3D_Solver_Optimized<<<gridDim, blockDim, 0, streams[gpu]>>>(
                            nx, dx, ny, dy, chunk_size, dz, dt, d_v_chunk, kernelData[i].d_pn, kernelData[i].d_pp);
                        break;
                }

                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaDeviceSynchronize();

                float timeElapsed = 0;
                cudaEventElapsedTime(&timeElapsed, start, stop);
                executionTimes[i][gpu][iter] = timeElapsed;
                //printf("GPU %d: %s execution time: %f ms\n", gpu, kernelData[i].name, timeElapsed);

                cudaMemcpy(kernelData[i].h_pp, kernelData[i].d_pp, nx * ny * chunk_size * sizeof(float), cudaMemcpyDeviceToHost);

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
                    fwrite(kernelData[i].h_pp, sizeof(float), nx * ny * chunk_size, resultFile);
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




int main(int argc, char **argv) {
    /// Set problem size
    int nx = NX;
    int ny = NY;
    int nz = NZ;
    int size = nx * ny * nz;

    /// Set simulation parameters
    float dx = DX;
    float dy = DY;
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
    float *h_v = (float*)malloc(nx * ny * nz * sizeof(float));
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
    for (int i=0; i < nx * ny * nz; i++) {
    	// h_pn1[i] = h_pn2[i] = h_pn3[i] = rand() / (float)RAND_MAX;
 	    //h_pn_Galerkin[i] = h_pn_Leapfrog[i] = h_pn_CrankNicolson[i] = h_pn_ADI[i] =
	    //h_pn_Sigma[i] = h_pn_LaxWendroff[i] = h_pn_FractionalStep[i] = h_pn_TVD[i] =
	    // h_pn_MacCormack[i] = h_pn_PSOR[i] = h_pn_FVS[i] = rand() / (float)RAND_MAX;

	    h_pn_Galerkin[i] = h_pn_Galerkin_Optimized[i] =
	    h_pn_Leapfrog[i] = h_pn_Leapfrog_Optimized[i] =
	    h_pn_CrankNicolson[i] = h_pn_CrankNicolson_Optimized[i] =
	    h_pn_ADI[i] = h_pn_ADI_Optimized[i] =
	    h_pn_Sigma[i] = h_pn_Sigma_Optimized[i] =
	    h_pn_LaxWendroff[i] =  h_pn_LaxWendroff_Optimized[i] =
	    h_pn_FractionalStep[i] = h_pn_FractionalStep_Optimized[i] =
	    h_pn_TVD[i] = h_pn_TVD_Optimized[i] =
	    h_pn_MacCormack[i] = h_pn_MacCormack_Optimized[i] =
	    h_pn_PSOR[i] = h_pn_PSOR_Optimized[i] =
	    h_pn_FVS[i] = h_pn_FVS_Optimized[i] = 1 + rand() % 1000;

	    //h_pn_Galerkin[i] = h_pn_Leapfrog[i] = h_pn_CrankNicolson[i] = h_pn_ADI[i] =
	    //h_pn_Sigma[i] = h_pn_LaxWendroff[i] = h_pn_FractionalStep[i] = h_pn_TVD[i] =
	    //h_pn_MacCormack[i] = h_pn_PSOR[i] = h_pn_FVS[i] = -1.0f + 2.0f * rand() / (float)RAND_MAX;
	    // h_pn1[i] = h_pn2[i] = h_pn3[i] = -1.0f + 2.0f * rand() / (float)RAND_MAX;
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
    cudaMemcpy(d_pn_Galerkin, h_pn_Galerkin, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_Galerkin_Optimized, h_pn_Galerkin_Optimized, size*sizeof(float),cudaMemcpyHostToDevice);
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
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x);

    dim3 blockSize2(8, 8);
    dim3 gridSize2((nx + blockSize2.x - 1) / blockSize2.x);

    /// Current block size
    int currentBlockSize = blockSize.x * blockSize.y * blockSize.z;
    int blockSizeLimit;
    cudaDeviceGetAttribute(&blockSizeLimit, cudaDevAttrMaxThreadsPerBlock,0);
    printf("Max Threads Per Block: %d\n", blockSizeLimit);

    /// Check the block size and adjust it if it exceeds the limit
    if (currentBlockSize > blockSizeLimit) {
    	printf("The block size exceeds the GPU limit, changing the block or grid size..\n");

	    /// Change the block or grid size according to GPU limits
	    blockSize.x /= 2;
	    blockSize.y /= 2;
	    blockSize.z /= 2;

	    gridSize.x = (nx + blockSize.x - 1) / blockSize.x;
	    gridSize.y = (ny + blockSize.y - 1) / blockSize.y;
	    gridSize.z = (nz + blockSize.z - 1) / blockSize.z;
    }


    ///==========================================================
    /// Launch Discontinuous Galerkin kernel and measure time
    ///==========================================================
/*
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    Discontinuous_Galerkin_3D_Solver<<<gridSize, blockSize>>>(nx, dx, ny, dy, nz, dz, dt, d_v, d_pn_Galerkin, d_pp_Galerkin);
    checkCUDAError("3D Galerkin Kernel launch");

    cudaEventRecord(stop1);
    cudaDeviceSynchronize();

    float time1 = 0;
    cudaEventElapsedTime(&time1, start1, stop1);
    printf("Total Execution Time on GPU for 3D Discontinuous Galerkin kernel: %f ms\n", time1);

    // Transfer the result of Discontinuous_Galerkin_3D_Solver from device to host
    cudaMemcpy(h_pp_Galerkin, d_pp_Galerkin, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the Galerkin_time array from device to host
    float GalerkinTime[N];
    cudaMemcpyFromSymbol(GalerkinTime, Galerkin_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of Galerkin elapsed time to a file1
    FILE *file1 = fopen("GalerkinTime_3D_data.txt", "w");
    if (file1 == NULL) {
    	fprintf(stderr, "Error opening GalerkinTime_3D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file1, "%.6f\n", GalerkinTime[i]);
    }
    fclose(file1);
*/
/*
    // Save the result of Galerkin_3D_solver to a file_a
    FILE *file_a = fopen("Galerkin3DSolver.txt", "w");
    if (file_a == NULL) {
	    fprintf(stderr, "Error opening Galerkin3DSolver.txt file..\n");
	    return 1;
    }
    for (int j=0; j<size; j++) {
	    fprintf(file_a, "%.6f\n", h_pp_Galerkin[j]);
    }
    fclose(file_a);
*/


    ///==========================================================
    /// Launch all kernels and measure time
    ///==========================================================

    Measure_And_Execute_Kernels(nx, ny, nz, dx, dy, dz, dt, d_v,
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





