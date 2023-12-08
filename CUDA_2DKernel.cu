#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cuda.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define N 1024

#define NX 128
#define NZ 128

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

void checkCUDAError(const char *message) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA Error: %s: %s.\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ float Galerkin_time[N];
__device__ float Leapfrog_time[N];
__device__ float CrankNicolson_time[N];
__device__ float ADI_time[N];
__device__ float Sigma_time[N];
__device__ float LaxWendroff_time[N];
__device__ float FractionalStep_time[N];
__device__ float MacCormack_time[N];
__device__ float TVD_time[N];
__device__ float PSOR_time[N];
__device__ float FVS_time[N];


__global__ void Galerkin_2D_Solver(
    int nx, float dx,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start imt
    clock_t start_time = clock();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockDim.y * blockDim.y + threadIdx.y;

    // Ensure the threads are within the grid size
    if (i < nx && j < nz) {
        int idx = i + j * nx;

        // Thread-local input and output arrays
        __shared__ float r_pn[p_NF];    // thread-local input
        __shared__ float r_pp[p_NF];    // thread-locak output

        // Shared memory arrays for second derivatives
        __shared__ float s_d2px[p_TM][p_TM];
        __shared__ float s_d2pz[p_TM][p_TM];

        // Load pressure field per thread memory
        PlaneThreads2D {
            const int idxl = i * p_NF + j * p_TM;
            #pragma unroll
            for (int n = 0; n < p_NF; n++) {
                r_pn[n] = d_pn[idxl + n];
                r_pp[n] = 0.0f;
            }
        }
        __syncthreads();

        // Calculate second derivatives
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

        // compute the wave equation
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

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("Galerkin Execution time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    Galerkin_time[i] = elapsed_time;
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
                            zscale * (infront3 + behind2) +
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


int main (int argc, char **argv) {
    // Set problem size
    int nx = NX;
    int nz = NZ;
    int size = nx * nz;

    // Set simulation parameters
    float dx = DX;
    float dz = DZ;
    float dt = DT;

    // Allocate memory on the host
    float *h_pn_Galerkin = (float*)malloc(size * sizeof(float));
    float *h_pn_Leapfrog = (float*)malloc(size * sizeof(float));
    float *h_pn_CrankNicolson = (float*)malloc(size * sizeof(float));
    float *h_pn_ADI = (float*)malloc(size * sizeof(float));
    float *h_pn_Sigma = (float*)malloc(size * sizeof(float));
    float *h_pn_LaxWendroff = (float*)malloc(size * sizeof(float));
    float *h_pn_FractionalStep = (float*)malloc(size * sizeof(float));
    float *h_pn_MacCormack = (float*)malloc(size * sizeof(float));
    float *h_pn_TVD = (float*)malloc(size * sizeof(float));
    float *h_pn_PSOR = (float*)malloc(size * sizeof(float));
    float *h_pn_FVS = (float*)malloc(size * sizeof(float));
    float *h_v = (float*)malloc(size * sizeof(float));
    float *h_pp_Galerkin = (float*)malloc(size * sizeof(float));
    float *h_pp_Leapfrog = (float*)malloc(size * sizeof(float));
    float *h_pp_CrankNicolson = (float*)malloc(size * sizeof(float));
    float *h_pp_ADI = (float*)malloc(size * sizeof(float));
    float *h_pp_Sigma = (float*)malloc(size * sizeof(float));
    float *h_pp_LaxWendroff = (float*)malloc(size * sizeof(float));
    float *h_pp_FractionalStep = (float*)malloc(size * sizeof(float));
    float *h_pp_MacCormack = (float*)malloc(size * sizeof(float));
    float *h_pp_TVD = (float*)malloc(size * sizeof(float));
    float *h_pp_PSOR = (float*)malloc(size * sizeof(float));
    float *h_pp_FVS = (float*)malloc(size * sizeof(float));

    // Initialize input data with random values
#pragma unroll
    for (int i=0; i < size; i++) {
	    h_pn_Galerkin[i] = h_pn_Leapfrog[i] = h_pn_CrankNicolson[i] = h_pn_ADI[i] =
	    h_pn_Sigma[i] = h_pn_LaxWendroff[i] = h_pn_FractionalStep[i] = h_pn_TVD[i] =
	    h_pn_MacCormack[i] = h_pn_PSOR[i] = h_pn_FVS[i] = 1 + rand() % 1000;
    }

    // Allocate memory on the device
    float *d_pn_Galerkin, *d_pn_Leapfrog, *d_pn_CrankNicolson, *d_pn_ADI;
    float *d_pn_Sigma, *d_pn_LaxWendroff, *d_pn_FractionalStep, *d_pn_MacCormack;
    float *d_pn_TVD, *d_pn_PSOR, *d_pn_FVS;
    float *d_v;
    float *d_pp_Galerkin, *d_pp_Leapfrog, *d_pp_CrankNicolson, *d_pp_ADI;
    float *d_pp_Sigma, *d_pp_LaxWendroff, *d_pp_FractionalStep, *d_pp_MacCormack;
    float *d_pp_TVD, *d_pp_PSOR, *d_pp_FVS;
    cudaMalloc((void**)&d_pn_Galerkin, size * sizeof(float));
    cudaMalloc((void**)&d_pn_Leapfrog, size * sizeof(float));
    cudaMalloc((void**)&d_pn_CrankNicolson, size * sizeof(float));
    cudaMalloc((void**)&d_pn_ADI, size * sizeof(float));
    cudaMalloc((void**)&d_pn_Sigma, size * sizeof(float));
    cudaMalloc((void**)&d_pn_LaxWendroff, size * sizeof(float));
    cudaMalloc((void**)&d_pn_FractionalStep, size * sizeof(float));
    cudaMalloc((void**)&d_pn_MacCormack, size * sizeof(float));
    cudaMalloc((void**)&d_pn_TVD, size * sizeof(float));
    cudaMalloc((void**)&d_pn_PSOR, size * sizeof(float));
    cudaMalloc((void**)&d_pn_FVS, size * sizeof(float));
    cudaMalloc((void**)&d_v, size * sizeof(float));
    cudaMalloc((void**)&d_pp_Galerkin, size * sizeof(float));
    cudaMalloc((void**)&d_pp_Leapfrog, size * sizeof(float));
    cudaMalloc((void**)&d_pp_CrankNicolson, size * sizeof(float));
    cudaMalloc((void**)&d_pp_ADI, size * sizeof(float));
    cudaMalloc((void**)&d_pp_Sigma, size * sizeof(float));
    cudaMalloc((void**)&d_pp_LaxWendroff, size * sizeof(float));
    cudaMalloc((void**)&d_pp_FractionalStep, size * sizeof(float));
    cudaMalloc((void**)&d_pp_MacCormack, size * sizeof(float));
    cudaMalloc((void**)&d_pp_TVD, size * sizeof(float));
    cudaMalloc((void**)&d_pp_PSOR, size * sizeof(float));
    cudaMalloc((void**)&d_pp_FVS, size * sizeof(float));

    // Transfer input data from host to device
    cudaMemcpy(d_pn_Galerkin, h_pn_Galerkin, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_Leapfrog, h_pn_Leapfrog, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_CrankNicolson, h_pn_CrankNicolson, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_ADI, h_pn_ADI, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_Sigma, h_pn_Sigma, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_LaxWendroff, h_pn_LaxWendroff, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_FractionalStep, h_pn_FractionalStep, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_MacCormack, h_pn_MacCormack, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_TVD, h_pn_TVD, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_PSOR, h_pn_PSOR, size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn_FVS, h_pn_FVS, size*sizeof(float),cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x);

    dim3 blockSize2(8);
    dim3 gridSize2((nx + blockSize2.x - 1) / blockSize2.x);

    // Current block size
    // int currentBlockSize = blockSize.x * blockSize.y * blockSize.z;
    int blockSizeLimit;
    cudaDeviceGetAttribute(&blockSizeLimit, cudaDevAttrMaxThreadsPerBlock,0);
    printf("Max Threads Per Block: %d\n", blockSizeLimit);

    //=============================================================
    // Launch Galerkin Kernel and measure time
    //=============================================================
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

    // Transfer the result of Discontinuous_Galerkin_2D_Solver from device to host
    cudaMemcpy(h_pp_Galerkin, d_pp_Galerkin, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the Galerkin_time array from device to host
    float GalerkinTime[N];
    cudaMemcpyFromSymbol(GalerkinTime, Galerkin_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of Galerkin elapsed time to a file1
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
    // Save the result of Galerkin_2D_solver to a file_a
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

    //=============================================================
    // Launch Leapfrog Kernel and measure time
    //=============================================================

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    Leapfrog_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_Leapfrog, d_pp_Leapfrog);
    checkCUDAError("2D Leapfrog Kernel launch");

    cudaEventRecord(stop2);
    cudaDeviceSynchronize();

    float time2 = 0;
    cudaEventElapsedTime(&time2, start2, stop2);
    printf("Total Execution Time on GPU for 2D Leapfrog kernel: %f ms\n", time2);

    // Transfer the result of Leapfrog_2D_solver from device to host
    cudaMemcpy(h_pp_Leapfrog, d_pp_Leapfrog, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the Leapfrog_time array from device to host
    float LeapfrogTime[N];
    cudaMemcpyFromSymbol(LeapfrogTime, Leapfrog_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of Leapfrog elapsed time to a file1
    FILE *file2 = fopen("LeapfrogTime_2D_data.txt", "w");
    if (file2 == NULL) {
    	fprintf(stderr, "Error opening LeapfrogTime_2D_data.txt file..\n");
	return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file2, "%.6f\n", LeapfrogTime[i]);
    }
    fclose(file2);
/*
    // Save the result of Leapfrog_2D_solver to a file_b
    FILE *file_b = fopen("Leapfrog2DSolver.txt", "w");
    if (file_b == NULL) {
	    fprintf(stderr, "Error opening Leapfrog2DSolver.txt file..\n");
	    return 1;
    }
    for (int k=0; k<size; k++) {
	    fprintf(file_b, "%.6f\n", h_pp_Leapfrog[k]);
    }
    fclose(file_b);
*/

    //=========================================================
    // Launch Crank-Nicolson kernel and measure time
    //=========================================================
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);

    CrankNicolson_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_CrankNicolson, d_pp_CrankNicolson);
    checkCUDAError("2D CrankNicolson Kernel launch");

    cudaEventRecord(stop3);
    cudaDeviceSynchronize();

    float time3 = 0;
    cudaEventElapsedTime(&time3, start3, stop3);
    printf("Total Execution Time on GPU for 2D CrankNicolson kernel: %f ms\n", time3);

    // Transfer the result of CrankNicolson_2D_solver from device to host
    cudaMemcpy(h_pp_CrankNicolson, d_pp_CrankNicolson, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the CrankNicolson_time array from device to host
    float CrankNicolsonTime[N];
    cudaMemcpyFromSymbol(CrankNicolsonTime, CrankNicolson_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of CrankNicolson elapsed time to a file3
    FILE *file3 = fopen("CrankNicolsonTime_2D_data.txt", "w");
    if (file3 == NULL) {
    	fprintf(stderr, "Error opening CranknicolsonTime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file3, "%.6f\n", CrankNicolsonTime[i]);
    }
    fclose(file3);
/*
    // Save the result of CrankNicolson_2D_solver to a file_c
    FILE *file_c = fopen("CrankNicolson2DSolver.txt", "w");
    if (file_c == NULL) {
	    fprintf(stderr, "Error opening CrankNicolson2DSolver.txt file..\n");
	    return 1;
    }
    for (int k=0; k<size; k++) {
	    fprintf(file_c, "%.6f\n", h_pp_CrankNicolson[k]);
    }
    fclose(file_c);
*/


    //=========================================================
    // Launch ADI kernel and measure time
    //=========================================================
    cudaEvent_t start4, stop4;
    cudaEventCreate(&start4);
    cudaEventCreate(&stop4);
    cudaEventRecord(start4);

    ADI_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_ADI, d_pp_ADI);
    checkCUDAError("2D ADI Kernel launch");

    cudaEventRecord(stop4);
    cudaDeviceSynchronize();

    float time4 = 0;
    cudaEventElapsedTime(&time4, start4, stop4);
    printf("Total Execution Time on GPU for 2D ADI kernel: %f ms\n", time4);

    // Transfer the result of ADI_2D_solver from device to host
    cudaMemcpy(h_pp_ADI, d_pp_ADI, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the ADI_time array from device to host
    float ADITime[N];
    cudaMemcpyFromSymbol(ADITime, ADI_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of ADI elapsed time to a file4
    FILE *file4 = fopen("ADITime_2D_data.txt", "w");
    if (file4 == NULL) {
    	fprintf(stderr, "Error opening ADITime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file4, "%.6f\n", ADITime[i]);
    }
    fclose(file4);
/*
    // Save the result of ADI_2D_solver to a file_d
    FILE *file_d = fopen("ADI2DSolver.txt", "w");
    if (file_d == NULL) {
	    fprintf(stderr, "Error opening ADI2DSolver.txt file..\n");
	    return 1;
    }
    for (int k=0; k<size; k++) {
	    fprintf(file_d, "%.6f\n", h_pp_ADI[k]);
    }
    fclose(file_d);
*/

    //=========================================================
    // Launch Sigma kernel and measure time
    //=========================================================
    cudaEvent_t start5, stop5;
    cudaEventCreate(&start5);
    cudaEventCreate(&stop5);
    cudaEventRecord(start5);

    Sigma_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_Sigma, d_pp_Sigma);
    checkCUDAError("2D Sigma Kernel launch");

    cudaEventRecord(stop5);
    cudaDeviceSynchronize();

    float time5 = 0;
    cudaEventElapsedTime(&time5, start5, stop5);
    printf("Total Execution Time on GPU for 2D Sigma kernel: %f ms\n", time5);

    // Transfer the result of Sigma_2D_solver from device to host
    cudaMemcpy(h_pp_Sigma, d_pp_Sigma, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the Sigma_time array from device to host
    float SigmaTime[N];
    cudaMemcpyFromSymbol(SigmaTime, Sigma_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of Sigma elapsed time to a file5
    FILE *file5 = fopen("SigmaTime_2D_data.txt", "w");
    if (file5 == NULL) {
    	fprintf(stderr, "Error opening SigmaTime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file5, "%.6f\n", SigmaTime[i]);
    }
    fclose(file5);
/*
    // Save the result of Sigma_2D_solver to a file_e
    FILE *file_e = fopen("Sigma2DSolver.txt", "w");
    if (file_e == NULL) {
	    fprintf(stderr, "Error opening Sigma2DSolver.txt file..\n");
	    return 1;
    }
    for (int k=0; k<size; k++) {
	    fprintf(file_e, "%.6f\n", h_pp_Sigma[k]);
    }
    fclose(file_e);
*/

    //=========================================================
    // Launch LaxWendroff kernel and measure time
    //=========================================================
    cudaEvent_t start6, stop6;
    cudaEventCreate(&start6);
    cudaEventCreate(&stop6);
    cudaEventRecord(start6);

    LaxWendroff_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_LaxWendroff, d_pp_LaxWendroff);
    checkCUDAError("2D LaxWendroff Kernel launch");

    cudaEventRecord(stop6);
    cudaDeviceSynchronize();

    float time6 = 0;
    cudaEventElapsedTime(&time6, start6, stop6);
    printf("Total Execution Time on GPU for 2D LaxWendroff kernel: %f ms\n", time6);

    // Transfer the result of LaxWendroff_2D_solver from device to host
    cudaMemcpy(h_pp_LaxWendroff, d_pp_LaxWendroff, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the LaxWendroff_time array from device to host
    float LaxWendroffTime[N];
    cudaMemcpyFromSymbol(LaxWendroffTime, LaxWendroff_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of LaxWendroff elapsed time to a file6
    FILE *file6 = fopen("LaxWendroffTime_2D_data.txt", "w");
    if (file6 == NULL) {
    	fprintf(stderr, "Error opening LaxWendroffTime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file6, "%.6f\n", LaxWendroffTime[i]);
    }
    fclose(file6);
/*
    // Save the result of LaxWendroff_2D_solver to a file_f
    FILE *file_f = fopen("LaxWendroff2DSolver.txt", "w");
    if (file_f == NULL) {
	    fprintf(stderr, "Error opening LaxWendroff2DSolver.txt file..\n");
	    return 1;
    }
    for (int k=0; k<size; k++) {
	    fprintf(file_f, "%.6f\n", h_pp_LaxWendroff[k]);
    }
    fclose(file_f);
*/

    //=========================================================
    // Launch Fractional Step kernel and measure time
    //=========================================================
    cudaEvent_t start7, stop7;
    cudaEventCreate(&start7);
    cudaEventCreate(&stop7);
    cudaEventRecord(start7);

    FractionalStep_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_FractionalStep, d_pp_FractionalStep);
    checkCUDAError("2D FractionalStep Kernel launch");

    cudaEventRecord(stop7);
    cudaDeviceSynchronize();

    float time7 = 0;
    cudaEventElapsedTime(&time7, start7, stop7);
    printf("Total Execution Time on GPU for 2D FractionalStep kernel: %f ms\n", time7);

    // Transfer the result of FractionalStep_2D_solver from device to host
    cudaMemcpy(h_pp_FractionalStep, d_pp_FractionalStep, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the FractionalStep_time array from device to host
    float FractionalStepTime[N];
    cudaMemcpyFromSymbol(FractionalStepTime, FractionalStep_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of FractionalStep elapsed time to a file7
    FILE *file7 = fopen("FractionalStepTime_2D_data.txt", "w");
    if (file7 == NULL) {
    	fprintf(stderr, "Error opening FractionalStepTime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file7, "%.6f\n", FractionalStepTime[i]);
    }
    fclose(file7);
/*
    // Save the result of FractionalStep_2D_solver to a file_g
    FILE *file_g = fopen("FractionalStep2DSolver.txt", "w");
    if (file_g == NULL) {
	    fprintf(stderr, "Error opening FractionalStep2DSolver.txt file..\n");
	    return 1;
    }
    for (int k=0; k<size; k++) {
	    fprintf(file_g, "%.6f\n", h_pp_FractionalStep[k]);
    }
    fclose(file_g);
*/

    //=========================================================
    // Launch MacCormack kernel and measure time
    //=========================================================
    cudaEvent_t start8, stop8;
    cudaEventCreate(&start8);
    cudaEventCreate(&stop8);
    cudaEventRecord(start8);

    MacCormack_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_MacCormack, d_pp_MacCormack);
    checkCUDAError("2D MacCormack Kernel launch");

    cudaEventRecord(stop8);
    cudaDeviceSynchronize();

    float time8 = 0;
    cudaEventElapsedTime(&time8, start8, stop8);
    printf("Total Execution Time on GPU for 2D MacCormack kernel: %f ms\n", time8);

    // Transfer the result of MacCormack_2D_solver from device to host
    cudaMemcpy(h_pp_MacCormack, d_pp_MacCormack, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the MacCormack_time array from device to host
    float MacCormackTime[N];
    cudaMemcpyFromSymbol(MacCormackTime, MacCormack_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of MacCormack elapsed time to a file8
    FILE *file8 = fopen("MacCormackTime_2D_data.txt", "w");
    if (file8 == NULL) {
    	fprintf(stderr, "Error opening MacCormackTime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file8, "%.6f\n", MacCormackTime[i]);
    }
    fclose(file8);
/*
    // Save the result of MacCormack_2D_solver to a file_h
    FILE *file_h = fopen("MacCormack2DSolver.txt", "w");
    if (file_h == NULL) {
	    fprintf(stderr, "Error opening MacCormack2DSolver.txt file..\n");
	    return 1;
    }
    for (int k=0; k<size; k++) {
	    fprintf(file_h, "%.6f\n", h_pp_MacCormack[k]);
    }
    fclose(file_h);
*/

    //=========================================================
    // Launch TVD kernel and measure time
    //=========================================================
    cudaEvent_t start9, stop9;
    cudaEventCreate(&start9);
    cudaEventCreate(&stop9);
    cudaEventRecord(start9);

    TVD_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_TVD, d_pp_TVD);
    checkCUDAError("2D TVD Kernel launch");

    cudaEventRecord(stop9);
    cudaDeviceSynchronize();

    float time9 = 0;
    cudaEventElapsedTime(&time9, start9, stop9);
    printf("Total Execution Time on GPU for 2D TVD kernel: %f ms\n", time9);

    // Transfer the result of TVD_2D_solver from device to host
    cudaMemcpy(h_pp_TVD, d_pp_TVD, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the TVD_time array from device to host
    float TVDTime[N];
    cudaMemcpyFromSymbol(TVDTime, TVD_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of TVD elapsed time to a file1
    FILE *file9 = fopen("TVDTime_2D_data.txt", "w");
    if (file9 == NULL) {
    	fprintf(stderr, "Error opening TVDTime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file9, "%.6f\n", TVDTime[i]);
    }
    fclose(file9);
/*
    // Save the result of TVD_2D_solver to a file_c
    FILE *file_i = fopen("TVDSolver.txt", "w");
    if (file_i == NULL) {
	    fprintf(stderr, "Error opening TVD2DSolver.txt file..\n");
	    return 1;
    }
    for (int l=0; l<size; l++) {
	    fprintf(file_i, "%.6f\n", h_pp_TVD[l]);
    }
    fclose(file_i);
*/

    //=========================================================
    // Launch PSOR kernel and measure time
    //=========================================================
    cudaEvent_t start10, stop10;
    cudaEventCreate(&start10);
    cudaEventCreate(&stop10);
    cudaEventRecord(start10);

    PSOR_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_PSOR, d_pp_PSOR);
    checkCUDAError("2D PSOR Kernel launch");

    cudaEventRecord(stop10);
    cudaDeviceSynchronize();

    float time10 = 0;
    cudaEventElapsedTime(&time10, start10, stop10);
    printf("Total Execution Time on GPU for 2D PSOR kernel: %f ms\n", time10);

    // Transfer the result of PSOR_2D_solver from device to host
    cudaMemcpy(h_pp_PSOR, d_pp_PSOR, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the PSOR_time array from device to host
    float PSORTime[N];
    cudaMemcpyFromSymbol(PSORTime, PSOR_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of PSOR elapsed time to a file10
    FILE *file10 = fopen("PSORTime_2D_data.txt", "w");
    if (file10 == NULL) {
    	fprintf(stderr, "Error opening PSORTime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file10, "%.6f\n", PSORTime[i]);
    }
    fclose(file10);
/*
    // Save the result of PSOR_2D_solver to a file_h
    FILE *file_j = fopen("PSOR2DSolver.txt", "w");
    if (file_j == NULL) {
	    fprintf(stderr, "Error opening PSOR2DSolver.txt file..\n");
	    return 1;
    }
    for (int l=0; l<size; l++) {
	    fprintf(file_j, "%.6f\n", h_pp_PSOR[l]);
    }
    fclose(file_j);
*/

    //=========================================================
    // Launch FVS kernel and measure time
    //=========================================================
    cudaEvent_t start11, stop11;
    cudaEventCreate(&start11);
    cudaEventCreate(&stop11);
    cudaEventRecord(start11);

    FVS_2D_Solver<<<gridSize2, blockSize2>>>(nx, dx, nz, dz, dt, d_v, d_pn_FVS, d_pp_FVS);
    checkCUDAError("2D FVS Kernel launch");

    cudaEventRecord(stop11);
    cudaDeviceSynchronize();

    float time11 = 0;
    cudaEventElapsedTime(&time11, start11, stop11);
    printf("Total Execution Time on GPU for 2D FVS kernel: %f ms\n", time11);

    // Transfer the result of FVS_2D_solver from device to host
    cudaMemcpy(h_pp_FVS, d_pp_FVS, size*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the FVS_time array from device to host
    float FVSTime[N];
    cudaMemcpyFromSymbol(FVSTime, FVS_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of FVS elapsed time to a file10
    FILE *file11 = fopen("FVSTime_2D_data.txt", "w");
    if (file11 == NULL) {
    	fprintf(stderr, "Error opening FVSTime_2D_data.txt file..\n");
	    return 1;
    }
#pragma unroll
    for (int i=0; i < N; i++) {
    	fprintf(file11, "%.6f\n", FVSTime[i]);
    }
    fclose(file11);
/*
    // Save the result of FVS_2D_solver to a file_h
    FILE *file_k = fopen("FVS2DSolver.txt", "w");
    if (file_k == NULL) {
	    fprintf(stderr, "Error opening FVS2DSolver.txt file..\n");
	    return 1;
    }
    for (int l=0; l<size; l++) {
	    fprintf(file_k, "%.6f\n", h_pp_FVS[l]);
    }
    fclose(file_k);
*/

    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);
    cudaEventDestroy(start4);
    cudaEventDestroy(start5);
    cudaEventDestroy(start6);
    cudaEventDestroy(start7);
    cudaEventDestroy(start8);
    cudaEventDestroy(start9);
    cudaEventDestroy(start10);
    cudaEventDestroy(start11);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);
    cudaEventDestroy(stop3);
    cudaEventDestroy(stop4);
    cudaEventDestroy(stop5);
    cudaEventDestroy(stop6);
    cudaEventDestroy(stop7);
    cudaEventDestroy(stop8);
    cudaEventDestroy(stop9);
    cudaEventDestroy(stop10);
    cudaEventDestroy(stop11);

    free(h_pn_Galerkin);
    free(h_pn_Leapfrog);
    free(h_pn_CrankNicolson);
    free(h_pn_ADI);
    free(h_pn_Sigma);
    free(h_pn_LaxWendroff);
    free(h_pn_FractionalStep);
    free(h_pn_MacCormack);
    free(h_pn_TVD);
    free(h_pn_PSOR);
    free(h_pn_FVS);
    free(h_v);
    free(h_pp_Galerkin);
    free(h_pp_Leapfrog);
    free(h_pp_CrankNicolson);
    free(h_pp_ADI);
    free(h_pp_Sigma);
    free(h_pp_LaxWendroff);
    free(h_pp_FractionalStep);
    free(h_pp_MacCormack);
    free(h_pp_TVD);
    free(h_pp_PSOR);
    free(h_pp_FVS);

    //free(GalerkinTime);
    //free(LeapfrogTime);
    //free(TVDTime);

    cudaFree(d_pn_Galerkin);
    cudaFree(d_pn_Leapfrog);
    cudaFree(d_pn_CrankNicolson);
    cudaFree(d_pn_ADI);
    cudaFree(d_pn_Sigma);
    cudaFree(d_pn_LaxWendroff);
    cudaFree(d_pn_FractionalStep);
    cudaFree(d_pn_MacCormack);
    cudaFree(d_pn_TVD);
    cudaFree(d_pn_PSOR);
    cudaFree(d_pn_FVS);
    cudaFree(d_v);
    cudaFree(d_pp_Galerkin);
    cudaFree(d_pp_Leapfrog);
    cudaFree(d_pp_CrankNicolson);
    cudaFree(d_pp_ADI);
    cudaFree(d_pp_Sigma);
    cudaFree(d_pp_LaxWendroff);
    cudaFree(d_pp_FractionalStep);
    cudaFree(d_pp_MacCormack);
    cudaFree(d_pp_TVD);
    cudaFree(d_pp_PSOR);
    cudaFree(d_pp_FVS);

    return 0;
}






