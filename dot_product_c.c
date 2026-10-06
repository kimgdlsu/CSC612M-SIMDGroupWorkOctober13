#include "kernels.h"

void dot_product_c(int64_t n, const double* A, const double* B, double* Z)
{
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        sum += A[i] * B[i];
    }
    *Z = sum;
}