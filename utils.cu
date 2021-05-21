/**
 * @file utils.c
 * author Yujia Zhai (yzhai015@ucr.edu)
 * @brief 
 * @version 0.1
 * @date 2020-02-06
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <stdio.h>
#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

void print_vector(float *vec, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        //if (i % 5 == 0) printf("\n");
        printf(" %5.2f, ", vec[i]);
    }
    printf("\n");
}

void print_matrix_lda(float *A, int lda, int m, int n)
{
    int i,j;
    float *a_curr_pos=A;
    printf("[");
    for (i = 0; i < n; i++){
        for (j=0;j<m;j++){
            printf("%5.2f, ",*a_curr_pos);
            a_curr_pos++;
        }
        a_curr_pos+=(lda-m);
    }
    printf("]\n");
}

void print_matrix(const float *A, int m, int n)
{
    int i;
    printf("[");
    for (i = 0; i < m * n; i++)
    {

        if ((i + 1) % n == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % n == 0)
        {
            if (i + 1 < m * n)
                printf(";\n");
        }
    }
    printf("]\n");
}

double get_sec()
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec + 1e-6 * time.tv_usec);
}

void randomize_matrix(float *A, int m, int n)
{
    srand(time(NULL));
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i * n + j] = (float)(rand() % 100);
            if (rand() % 2 == 0)
            {
                A[i * n + j] *= -1.0;
            }
        }
    }
}

void copy_matrix(float *src, float *dest, int n)
{
    int i;
    for (i = 0; src + i && dest + i && i < n; i++)
    {
        *(dest + i) = *(src + i);
    }
    if (i != n)
    {
        printf("copy failed at %d while there are %d elements in total.\n", i, n);
    }
}

bool verify_matrix(float *mat1, float *mat2, int n)
{
    float diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < n; i++)
    {
        //diff = fabs(mat1[i] - mat2[i]) / fabs(mat2[i]);
        diff = fabs(mat1[i] - mat2[i]);
        if (diff > 1e-3) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i],mat2[i],i);
            return false;
        }
    }
    return true;
}