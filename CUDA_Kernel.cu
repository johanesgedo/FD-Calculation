#include <stdio.h>
#include <cstdio>
#include <cuda.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define NX 256
#define NY 256
#define NZ 256

#define DX 100
#define DY 100
#define DZ 100
#define DT 10

#define N 2048

#define BLOCK_DIMX 32
#define BLOCK_DIMY 32

#define p_TM 8
#define p_NF 8

#define cubeThreads 						\
    for (int k = threadIdx.z; k < p_TM; k += blockDim.z) 	\
        for (int j = threadIdx.y; j < p_TM; j += blockDim.y) 	\
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
__device__ float TVD_time[N];


// Discontinuous Galerkin method to solve 3D acoustic wave equation using OCCA algorithm
__global__ void Discontinuous_Galerkin_3D_Solver(
    int nx, float dx,
    int ny, float dy,
    int nz, float dz, float dt,
    float* __restrict__ d_v,
    float* __restrict__ d_pn,
    float* __restrict__ d_pp
)
{
    // Get the start time
    clock_t start_time = clock();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure that threads are within the grid size
    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;

        // Thread-local input and output arrays
        __shared__ float r_pn[p_NF]; // thread-local input
        __shared__ float r_pp[p_NF]; // thread-local output

        /// Shared memory arrays for second derivatives
        __shared__ float s_d2px[p_TM][p_TM][p_TM];
        __shared__ float s_d2py[p_TM][p_TM][p_TM];
        __shared__ float s_d2pz[p_TM][p_TM][p_TM];

        // Load pressure field per thread memory
        cubeThreads {
            const int idxl = i * p_NF + j * p_TM + k * p_TM * p_NF;
            #pragma unroll
            for (int n = 0; n < p_NF; ++n) {
                r_pn[n] = d_pn[idxl + n];
                r_pp[n] = 0.0f;
            }
        }
        __syncthreads();

        // Calculate second derivatives
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

        // Compute the wave equation
        cubeThreads {
            const int idxl = i * p_NF + j * p_TM + k * p_TM * p_NF;
            #pragma unroll
            for (int n = 0; n < p_NF; ++n) {
                r_pp[n] = d_v[idx] * d_v[idx] * (s_d2px[k][j][i] + s_d2py[k][j][i] + s_d2pz[k][j][i]) -
                                (r_pn[n] - 2.0f * d_pn[idxl + n]) / (dt * dt);
            }
        }
        __syncthreads();

        // Update the global residual memory
        cubeThreads {
            const int idxl = i * p_NF + j * p_TM + k * p_TM * p_NF;
            #pragma unroll
            for (int n = 0; n < p_NF; ++n) {
                d_pp[idxl + n] = r_pp[n];
            }
        }
    }

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("Discontinuous Galerkin Execution Time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    Galerkin_time[i] = elapsed_time;
}


// Leapfrog menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void Leapfrog_3D_Solver(
    int nx, float dx,       
    int ny, float dy,       
    int nz, float dz, float dt,       
    float* __restrict__ d_v,     
    float* __restrict__ d_pn,    
    float* __restrict__ d_pp     
)
{
    // Get the start time
    clock_t start_time = clock();

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

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("Leapfrog Execution Time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    Leapfrog_time[ix] = elapsed_time;
}


// Total Variation DIminishing (TVD) menthod to solve 3D acoustic wave equation using Micikevisius' algorithm
__global__ void TVD_3D_Solver(
    int nx, float dx,     
    int ny, float dy,   
    int nz, float dz, float dt,     
    float* __restrict__ d_v,   
    float* __restrict__ d_pn,   
    float* __restrict__ d_pp  
)
{
    // Get the start time
    clock_t start_time = clock();

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

    // Get the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    float elapsed_time = 1000.0 * (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the elapsed time
    // printf("TVD Execution Time: %f ms\n", elapsed_time);

    // Save the elapsed time to the global array
    TVD_time[ix] = elapsed_time;
}



int main(int argc, char **argv) {
    // Set problem size
    int nx = NX;
    int ny = NY;
    int nz = NZ;

    // Set simulation parameters
    float dx = DX;
    float dy = DY;
    float dz = DZ;
    float dt = DT;

    // Allocate memory on the host
    float *h_pn1 = (float*)malloc(nx * ny * nz * sizeof(float));
    float *h_pn2 = (float*)malloc(nx * ny * nz * sizeof(float));
    float *h_pn3 = (float*)malloc(nx * ny * nz * sizeof(float));
    float *h_v = (float*)malloc(nx * ny * nz * sizeof(float));
    float *h_pp1 = (float*)malloc(nx * ny * nz * sizeof(float));
    float *h_pp2 = (float*)malloc(nx * ny * nz * sizeof(float));
    float *h_pp3 = (float*)malloc(nx * ny * nz * sizeof(float));

    // Initialize input data with random values
    for (int i=0; i < nx * ny * nz; i++) {
    	// h_pn1[i] = h_pn2[i] = h_pn3[i] = rand() / (float)RAND_MAX;
	h_pn1[i] = h_pn2[i] = h_pn3[i] = 1 + rand() % 1000;
	// h_pn1[i] = h_pn2[i] = h_pn3[i] = -1.0f + 2.0f * rand() / (float)RAND_MAX;
    }

    // Allocate memory on the device
    float *d_pn1, *d_pn2, *d_pn3, *d_v, *d_pp1, *d_pp2, *d_pp3;
    cudaMalloc((void**)&d_pn1, nx*ny*nz * sizeof(float));
    cudaMalloc((void**)&d_pn2, nx*ny*nz * sizeof(float));
    cudaMalloc((void**)&d_pn3, nx*ny*nz * sizeof(float));
    cudaMalloc((void**)&d_v, nx*ny*nz * sizeof(float));
    cudaMalloc((void**)&d_pp1, nx*ny*nz * sizeof(float));
    cudaMalloc((void**)&d_pp2, nx*ny*nz * sizeof(float));
    cudaMalloc((void**)&d_pp3, nx*ny*nz * sizeof(float));

    // Transfer input data from host to device
    cudaMemcpy(d_pn1, h_pn1, nx*ny*nz*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn2, h_pn2, nx*ny*nz*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pn3, h_pn3, nx*ny*nz*sizeof(float),cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x);

    dim3 blockSize2(8, 8);
    dim3 gridSize2((nx + blockSize2.x - 1) / blockSize2.x);

    // Current block size
    int currentBlockSize = blockSize.x * blockSize.y * blockSize.z;
    int blockSizeLimit;
    cudaDeviceGetAttribute(&blockSizeLimit, cudaDevAttrMaxThreadsPerBlock,0);
    printf("Max Threads Per Block: %d\n", blockSizeLimit);

    // Check the block size and adjust it if it exceeds the limit
    if (currentBlockSize > blockSizeLimit) {
    	printf("The block size exceeds the GPU limit, changing the block or grid size..\n");
	
	// Change the block or grid size according to GPU limits
	blockSize.x /= 2;
	blockSize.y /= 2;
	blockSize.z /= 2;

	gridSize.x = (nx + blockSize.x - 1) / blockSize.x;
	gridSize.y = (ny + blockSize.y - 1) / blockSize.y;
	gridSize.z = (nz + blockSize.z - 1) / blockSize.z;
    }


    //==========================================================
    // Launch Discontinuous Galerkin kernel and measure time
    //==========================================================
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    
    Discontinuous_Galerkin_3D_Solver<<<gridSize, blockSize>>>(nx, dx, ny, dy, nz, dz, dt, d_v, d_pn1, d_pp1);
    checkCUDAError("Galerkin Kernel launch");
    
    cudaEventRecord(stop1);
    cudaDeviceSynchronize();

    float time1 = 0;
    cudaEventElapsedTime(&time1, start1, stop1);
    printf("Total Execution Time on GPU for Discontinuous Galerkin kernel: %f ms\n", time1);

    // Transfer the result of Discontinuous_Galerkin_3D_Solver from device to host
    cudaMemcpy(h_pp1, d_pp1, nx*ny*nz*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the Galerkin_time array from device to host
    float GalerkinTime[N];
    cudaMemcpyFromSymbol(GalerkinTime, Galerkin_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of Galerkin elapsed time to a file1
    FILE *file1 = fopen("GalerkinTime_data.txt", "w");
    if (file1 == NULL) {
    	fprintf(stderr, "Error opening GalerkinTime_data.txt file..\n");
	return 1;
    }
    for (int i=0; i < N; i++) {
    	fprintf(file1, "%.6f\n", GalerkinTime[i]);
    }
    fclose(file1);

    // Save the result of Galerkin_3D_solver to a file_a
    FILE *file_a = fopen("GalerkinSolver.txt", "w");
    if (file_a == NULL) {
	fprintf(stderr, "Error opening GalerkinSolver.txt file..\n");
	return 1;
    }
    for (int j=0; j<nx*ny*nz; j++) {
	fprintf(file_a, "%.6f\n", h_pp1[j]);
    }
    fclose(file_a);


    //=========================================================
    // Launch Leapfrog kernel and measure time
    //=========================================================
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    Leapfrog_3D_Solver<<<gridSize2, blockSize2>>>(nx, dx, ny, dy, nz, dz, dt, d_v, d_pn2, d_pp2);
    checkCUDAError("Leapfrog Kernel launch");

    cudaEventRecord(stop2);
    cudaDeviceSynchronize();

    float time2 = 0;
    cudaEventElapsedTime(&time2, start2, stop2);
    printf("Total Execution Time on GPU for Leapfrog kernel: %f ms\n", time2);

    // Transfer the result of Leapfrog_3D_solver from device to host
    cudaMemcpy(h_pp2, d_pp2, nx*ny*nz*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the Leapfrog_time array from device to host
    float LeapfrogTime[N];
    cudaMemcpyFromSymbol(LeapfrogTime, Leapfrog_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of Leapfrog elapsed time to a file1
    FILE *file2 = fopen("LeapfrogTime_data.txt", "w");
    if (file2 == NULL) {
    	fprintf(stderr, "Error opening LeapfrogTime_data.txt file..\n");
	return 1;
    }
    for (int i=0; i < N; i++) {
    	fprintf(file2, "%.6f\n", LeapfrogTime[i]);
    }
    fclose(file2);

    // Save the result of Leapfrog_3D_solver to a file_b
    FILE *file_b = fopen("LeapfrogSolver.txt", "w");
    if (file_b == NULL) {
	fprintf(stderr, "Error opening LeapfrogSolver.txt file..\n");
	return 1;
    }
    for (int k=0; k<nx*ny*nz; k++) {
	fprintf(file_b, "%.6f\n", h_pp2[k]);
    }
    fclose(file_b);


    //=========================================================
    // Launch TVD kernel and measure time
    //=========================================================
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);

    TVD_3D_Solver<<<gridSize2, blockSize2>>>(nx, dx, ny, dy, nz, dz, dt, d_v, d_pn3, d_pp3);
    checkCUDAError("TVD Kernel launch");

    cudaEventRecord(stop3);
    cudaDeviceSynchronize();

    float time3 = 0;
    cudaEventElapsedTime(&time3, start3, stop3);
    printf("Total Execution Time on GPU for TVD kernel: %f ms\n", time3);

    // Transfer the result of TVD_3D_solver from device to host
    cudaMemcpy(h_pp2, d_pp2, nx*ny*nz*sizeof(float),cudaMemcpyDeviceToHost);

    // Transfer the TVD_time array from device to host
    float TVDTime[N];
    cudaMemcpyFromSymbol(TVDTime, TVD_time, N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Save the result of Leapfrog elapsed time to a file1
    FILE *file3 = fopen("TVDTime_data.txt", "w");
    if (file3 == NULL) {
    	fprintf(stderr, "Error opening TVDTime_data.txt file..\n");
	return 1;
    }
    for (int i=0; i < N; i++) {
    	fprintf(file3, "%.6f\n", TVDTime[i]);
    }
    fclose(file3);

    // Save the result of TVD_3D_solver to a file_c
    FILE *file_c = fopen("TVDSolver.txt", "w");
    if (file_c == NULL) {
	fprintf(stderr, "Error opening TVDSolver.txt file..\n");
	return 1;
    }
    for (int l=0; l<nx*ny*nz; l++) {
	fprintf(file_c, "%.6f\n", h_pp3[l]);
    }
    fclose(file_c);


    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);
    cudaEventDestroy(stop3);

    free(h_pn1);
    free(h_pn2);
    free(h_pn3);
    free(h_v);
    free(h_pp1);
    free(h_pp2);
    free(h_pp3);

    //free(GalerkinTime);
    //free(LeapfrogTime);
    //free(TVDTime);

    cudaFree(d_pn1);
    cudaFree(d_pn2);
    cudaFree(d_pn3);
    cudaFree(d_v);
    cudaFree(d_pp1);
    cudaFree(d_pp2);
    cudaFree(d_pp3);

    return 0;
}



