#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "utils.h"
#include <unistd.h>

int main(int argc, char *argv[])
{
    if (argc != 4) {
        printf("please input [m] [n] [kernel_num].\n");
        printf("kernel_num == 1: mysgemv (default).\n");
        printf("kernel_num == 2: cuBLAS SGEMV.\n");
        exit(-1);
    }
    int m, n, kernel_num = 1;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    kernel_num = atoi(argv[3]);
    if (m != (m & -32) || n != (n & -32)) {
        printf("currently we only support m, n divisible by 32.\n");
        printf("rounded m, n to multipliers of 32.\n");
        m = (m & -32); n = (n & -32);
    }
    if ( (kernel_num!=1&&kernel_num!=2) || m <= 0 || n <= 0 ) {
        printf("Illegal input, returned.\n");
        exit(-1);
    }
    printf("m = %d, n = %d.\n", m, n);
    if (kernel_num == 1) printf("Testing my sgemv.\n");
    else printf("Testing cuBLAS SGEMV.\n");
    float *hA, *hX, *hY, *hY_ref;
    float *dA, *dX, *dY, *dY_ref;
    float elapsed_time;
    hA = (float*)malloc(sizeof(float) * m * n);
    hX = (float*)malloc(sizeof(float) * n);
    hY = (float*)malloc(sizeof(float) * m);
    hY_ref = (float*)malloc(sizeof(float) * m);
    float alpha = 1., beta = 1.;
    int N = 5;
    randomize_matrix(hA, m, n);
    randomize_matrix(hX, n, 1);
    randomize_matrix(hY, m, 1);
    randomize_matrix(hY_ref, m, 1);
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    CUDA_CALLER(cudaMalloc ((void**)&dA, sizeof(float) * m * n));
    CUDA_CALLER(cudaMalloc ((void**)&dX, sizeof(float) * n));
    CUDA_CALLER(cudaMalloc ((void**)&dY, sizeof(float) * m));
    CUDA_CALLER(cudaMalloc ((void**)&dY_ref, sizeof(float) * m));
    CUDA_CALLER(cudaMemcpy(dA, hA, sizeof(float) * m * n, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dX, hX, sizeof(float) * n, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dY, hY, sizeof(float) * m, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dY_ref, hY_ref, sizeof(float) * m, cudaMemcpyHostToDevice));
    cublasHandle_t myHandle; cublasCreate(&myHandle);

    if (kernel_num == 1){
        printf("Start the sanity check...\n");
        fflush(stdout);
        mysgemv(m, n, alpha, dA, m, dX, beta, dY);
        cublasSgemv(myHandle, CUBLAS_OP_N, m, n, &alpha, dA, m, dX, 1, &beta, dY_ref, 1);
        
        cudaDeviceSynchronize();
        cudaMemcpy(hY, dY, sizeof(float)*m, cudaMemcpyDeviceToHost);
        cudaMemcpy(hY_ref, dY_ref, sizeof(float)*m, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (!verify_matrix(hY, hY_ref, m)){
            printf("did not pass the sanity check, returned.\n");
            exit(-2);
        }else{
            printf("Sanity check passed. Start performance benchmarking...\n");
            fflush(stdout);
        }
    }

    cudaDeviceSynchronize();
    cudaEventRecord(beg);
    if (kernel_num == 1){
        for (int i = 0; i < N; i++){
            mysgemv(m, n, alpha, dA, m, dX, beta, dY);
        }
    }else{
        for (int i = 0; i < N; i++){
            cublasSgemv(myHandle, CUBLAS_OP_N, m, n, &alpha, dA, m, dX, 1, &beta, dY, 1);
        }
    }

    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.;
    printf("Average elasped time: %f second, performance: %f GFLOPS.\n", elapsed_time/N,2.*N*1e-9*m*n/elapsed_time);
    cudaDeviceSynchronize();
    free(hA);free(hX);free(hY);free(hY_ref);
    cudaFree(dA);cudaFree(dX);cudaFree(dY);cudaFree(dY_ref);
    cudaDeviceSynchronize();
    return 0;
}