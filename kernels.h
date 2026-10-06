#pragma once
/*
 * kernels.h  --  FROZEN common interface (owner: Person 1)
 *
 * Every kernel computes Z = sum_{i=0}^{n-1} A[i] * B[i]
 * and stores the result through the Z pointer.
 *
 * Windows x64 calling convention -> register mapping for the .asm kernels:
 *     RCX = n      (int64_t, number of elements)
 *     RDX = A      (const double*)
 *     R8  = B      (const double*)
 *     R9  = Z      (double*, output)
 *
 * Volatile (free to use without saving): RAX, RCX, RDX, R8-R11, XMM0-XMM5
 * Non-volatile (must save/restore if used): RBX, RBP, RDI, RSI, R12-R15, XMM6-XMM15
 *
 * Do NOT change these signatures without telling the whole group.
 */
#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

typedef void (*dot_kernel_fn)(int64_t n, const double* A, const double* B, double* Z);

/* Person 1 */
void dot_product_c(int64_t n, const double* A, const double* B, double* Z);
extern void dot_product_asm(int64_t n, const double* A, const double* B, double* Z);

/* Person 2 */
extern void dot_product_xmm(int64_t n, const double* A, const double* B, double* Z);

/* Person 3 */
extern void dot_product_ymm(int64_t n, const double* A, const double* B, double* Z);

#endif /* KERNELS_H */