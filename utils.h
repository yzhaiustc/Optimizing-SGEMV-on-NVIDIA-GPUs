#ifndef _UTIL_H_
#define _UTIL_H_
#include "sys/time.h"

#define CUDA_CALLER(call) do{\
  cudaError_t cuda_ret = (call);\
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    printf("  In the function call %s\n", #call);\
    exit(1);\
  }\
}while(0)
#define CUDA_KERNEL_CALLER(...) do{\
  if(cudaPeekAtLastError() != cudaSuccess){\
    printf("A CUDA error occurred prior to the kernel call %s at line %d\n", #__VA_ARGS__,  __LINE__); exit(1);\
  }\
  __VA_ARGS__;\
  cudaError_t cuda_ret = cudaPeekAtLastError();\
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    printf("  In the kernel call %s\n", #__VA_ARGS__);\
    exit(1);\
  }\
}while(0)

void randomize_matrix(float *A, int m, int n);
double get_sec();
void print_matrix(const float *A, int m, int n);
void print_vector(float *vec, int n);
void copy_matrix(float *src, float *dest, int n);
bool verify_matrix(float *mat1, float *mat2, int n);
void print_matrix_lda(float *A, int lda, int m, int n);
void mysgemv_gpu_v1(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y);
void mysgemv_gpu_v2(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y);
void mysgemv_gpu_v3(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y);
void mysgemv(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y);
#endif // _UTIL_H_