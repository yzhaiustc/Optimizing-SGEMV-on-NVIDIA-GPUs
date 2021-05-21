#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "utils.h"
#include <unistd.h>
// #define TILE_SIZE 64
#define DIM_M 32
#define CEIL_DIV(x, y) ((x)+(y)-1)/(y)
#define A(i,j) A[(i)+(j)*lda]
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
//v1 = s3 * v2 + v1
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
__global__
void sgemv_kernel_v1(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y){
    int tx = threadIdx.x, bx = blockIdx.x;
    A = &A(bx*DIM_M, 0);
    y = y + bx*DIM_M;
    float resY = 0.;
    for (int j = 0; j < n; j ++){
        resY += A(tx, j) * x[j];
    }
    y[tx] = alpha*resY + beta *y[tx];
}

void mysgemv_gpu_v1(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y){
    dim3 grid( CEIL_DIV(m, DIM_M), 1 );
    dim3 threads( DIM_M, 1 );
    sgemv_kernel_v1<<<grid, threads>>>(m, n, alpha, A, lda, x, beta, y);
}

__global__ 
void sgemv_kernel_v2(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y){
    int tx = threadIdx.x*4, bx = blockIdx.x, bd = blockDim.x*4;
    int tx1, tx2, tx3;
    tx1 = tx + 1; tx2 = tx + 2; tx3 = tx + 3;
    A = &A(bx*bd, 0);
    y = y + bx*bd;
    float resY[4], xj;
    memset(resY, 0, sizeof(resY));
    for (int j = 0; j < n; j ++){
        xj = x[j];
        resY[0] += A(tx, j) * xj;
        resY[1] += A(tx1, j) * xj;
        resY[2] += A(tx2, j) * xj;
        resY[3] += A(tx3, j) * xj;
    }
    y[tx] = alpha*resY[0] + beta *y[tx];
    y[tx1] = alpha*resY[1] + beta *y[tx1];
    y[tx2] = alpha*resY[2] + beta *y[tx2];
    y[tx3] = alpha*resY[3] + beta *y[tx3];
}

void mysgemv_gpu_v2(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y){
    dim3 grid( CEIL_DIV(m, 4*DIM_M), 1 );
    dim3 threads( DIM_M, 1 );
    sgemv_kernel_v2<<<grid, threads>>>(m, n, alpha, A, lda, x, beta, y);
}


__global__ 
void sgemv_kernel_v3(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y){
    int tx = (threadIdx.x<<2), bx = blockIdx.x, bd = (blockDim.x<<2);
    int bx_bd = bx*bd;
    A = &A(bx_bd, 0);
    y = y + bx_bd;
    float4 resY, vA, vY;
    float xj;
    resY.x=0.;resY.y=0.;resY.z=0.;resY.w=0.;
    #pragma unroll
    for (int j = 0; j < n; j ++){
        xj = x[j];
        vload(vA, &A(tx, j))
        vscal(resY, vA, xj)
    }
    vload(vY, y+tx)
    simd_axpby(resY, alpha, resY, beta, vY)
    vstore(y+tx, resY)
}

void mysgemv_gpu_v3(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y){
    dim3 grid( CEIL_DIV(m, 4*DIM_M), 1 );
    dim3 threads( DIM_M, 1 );
    sgemv_kernel_v3<<<grid, threads>>>(m, n, alpha, A, lda, x, beta, y);
}

void mysgemv(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y){
    if (m < 9216) mysgemv_gpu_v1(m, n, alpha, A, m, x, beta, y);
    else mysgemv_gpu_v3(m, n, alpha, A, m, x, beta, y);
}