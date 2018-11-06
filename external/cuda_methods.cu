/* cuda_methods.c

   ----------------------------------------------------------------------
   This file is part of ReAl-LiFE toolbox

   Copyright (C) (2018-) anonymized
   email: anonymized
   ----------------------------------------------------------------------
*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>     /* provides DBL_EPSILON */
#include <sys/types.h>

#define syncSize (32)
#define nThreads (32)

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ double ws_reduceResult(double val){
    volatile int tid = threadIdx.x;
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ void readMxEntries(unsigned long int* s_A, double* s_x, double* s_Val,
            const unsigned long int* A, const unsigned long int* F, const double* x,
            const double* Val, unsigned long int k, unsigned int nElem) {
    if(threadIdx.x < nElem) {
        *s_A = A[threadIdx.x+k];
        *s_Val = Val[threadIdx.x+k];
        *s_x = x[F[threadIdx.x+k]];
    }
}

__device__ void readMtEntries(unsigned long int* s_A, unsigned long int* s_F, double* s_Val,
            const unsigned long int* A, const unsigned long int* F, const double* Val,
            const unsigned long int k, unsigned int nElem) {
    if(threadIdx.x < nElem) {
        *s_A = A[threadIdx.x+k];
        *s_F = F[threadIdx.x+k];
        *s_Val = Val[threadIdx.x+k];
    }
}

__device__ void computeMx(double *l_mx, unsigned long int s_A, double s_x, double s_Val,
                    const double* D, unsigned int nElem) {
    int i;
    double x, val;
    unsigned long int a;
    for(int j=0; j<nElem; j++) {
        a = __shfl_sync(0xffffffff, s_A, j, 32);
        x = __shfl_sync(0xffffffff, s_x, j, 32);
        val = __shfl_sync(0xffffffff, s_Val, j, 32);
        for(i=0; i<nTh; i++) {
            l_mx[i] += D[a+threadIdx.x+i*32]*x*val;
        }
    }
}

__device__ void computeMtx(double *s_out, const unsigned long int s_A, double s_Val, double* l_mx,
                    const double* D, unsigned int nElem) {
    volatile double res;
    unsigned long int a;
    double val;
    for(int j=0; j<nElem; j++) {
        a = __shfl_sync(0xffffffff, s_A, j, 32);
        val = __shfl_sync(0xffffffff, s_Val, j, 32);
        res = 0;
        for(int i=0; i<nTh; i++) {
            res +=  l_mx[i] * D[a+threadIdx.x+i*32]*val;
        }
        res = ws_reduceResult(res);
        if(threadIdx.x == 0){
            s_out[j] = res;
        }
    }
}

__global__ void M_times(double* mx, const unsigned long int* A,
        const unsigned long int* V, const unsigned long int* F,
        const double* Val, const double* D, const double* x,
        const unsigned int nTheta, const unsigned long int nVoxels,
        const unsigned long int nCoeffs, const double* voxelBounds)
{
    __shared__ unsigned long int s_v;
    unsigned long int l_A;
    double l_x;
    double l_Val;

    unsigned long int cStart = voxelBounds[blockIdx.x];
    unsigned long int cEnd = voxelBounds[blockIdx.x+1];

    unsigned long int k;
    unsigned int nElem;
    double l_mx[nTh];

    if(threadIdx.x==0){
        s_v = V[cStart];
    }
    __syncthreads();

    for(int i=0; i<nTh; i++){
        l_mx[i] = 0;
    }

    k = cStart;
    while(k < cEnd) {
        /* Read N next entries */
        nElem = syncSize;
        if(k + syncSize > cEnd) nElem = cEnd - k;
        __syncthreads();
        readMxEntries(&l_A, &l_x, &l_Val, A, F, x, Val, k, nElem);
        __syncthreads();
        
        /* Compure Mx */
        computeMx(l_mx, l_A, l_x, l_Val, D, nElem);
        k = k + nElem;
    }
    for(int i=0; i<nTh; i++){
        mx[s_v+threadIdx.x+i*32] = l_mx[i];
    }
}

__global__ void Mtransp_times(double* mtx, const unsigned long int* A,
        const unsigned long int* V, const unsigned long int* F,
        const double* Val, const double* D, const double* x,
        const unsigned int nTheta, const unsigned long int nVoxels,
        const unsigned long int nCoeffs, const double* voxelBounds)
{
    __shared__ double s_out[syncSize];
    __shared__ unsigned long int s_v;

    unsigned long int l_a, l_f;
    double l_val;

    unsigned long int cStart = voxelBounds[blockIdx.x];
    unsigned long int cEnd = voxelBounds[blockIdx.x+1];

    unsigned long int k;
    unsigned int nElem;
    if(threadIdx.x==0){
        s_v = V[cStart];
    }
    __syncthreads();

    double l_mx[nTh];
    for(int i=0; i<nTh; i++){
        l_mx[i] = x[s_v + threadIdx.x + i*32];
    }

    k = cStart;
    while(k < cEnd) {
        /* Read N next entries */
        nElem = syncSize;
        if(k + syncSize > cEnd) nElem = cEnd - k;
        __syncthreads();
        readMtEntries(&l_a, &l_f, &l_val, A, F, Val, k, nElem);
        __syncthreads();

        /* Compute Mtransp x*/
        computeMtx(s_out, l_a, l_val, l_mx, D, nElem);
        __syncthreads();

        /* Write back the results */
        if(threadIdx.x < nElem){ //0.007-0.009
            atomicAdd(&mtx[l_f], s_out[threadIdx.x]); 
        }
        k = k + nElem;
    } 
    return;
}
