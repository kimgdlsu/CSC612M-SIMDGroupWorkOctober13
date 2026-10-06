/*
 * main.c  --  shared driver (owner: Person 1; others should not edit)
 *
 * Responsibilities:
 *   - allocate + initialize A and B
 *   - run every kernel RUNS times per size, time the kernel call only
 *   - verify each kernel against the C result
 *   - print results (n, kernel, Z, avg time, PASS/FAIL)
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <malloc.h>     /* _aligned_malloc / _aligned_free */
#include <windows.h>    /* QueryPerformanceCounter */

#include "kernels.h"

 /* ---------------- configuration ---------------- */

#define RUNS        30
#define ALIGNMENT   32          /* 32-byte alignment suits YMM loads */
#define REL_TOL     1e-9        /* see check_result() */

#ifdef _DEBUG
#define BUILD_NAME "DEBUG"
#else
#define BUILD_NAME "RELEASE"
#endif

/* Benchmark sizes. Lower the last one to (1LL << 29) or (1LL << 28)
 * if allocation fails (2^30 doubles = 8 GB per vector). */
static const int64_t BENCH_SIZES[] = {
    1LL << 20,
    1LL << 26,
    1LL << 28,              /* TODO: try 1LL << 30 first, document what fits */
};

/* Boundary sizes (item 3a): not multiples of 2 or 4, to exercise tail code. */
static const int64_t BOUNDARY_SIZES[] = {
    1, 2, 3, 5, 7, 1003,
    (1LL << 20) + 1,
    (1LL << 20) + 3,
};

typedef struct {
    const char* name;
    dot_kernel_fn fn;
} kernel_entry;

/* Index 0 MUST be the C version: it is the reference for all others. */
static const kernel_entry KERNELS[] = {
    { "C",          dot_product_c   },
    { "x86-64 ASM", dot_product_asm },
    { "SIMD XMM",   dot_product_xmm },   /* stub until Person 2 delivers */
    { "SIMD YMM",   dot_product_ymm },   /* stub until Person 3 delivers */
};
#define NUM_KERNELS (sizeof(KERNELS) / sizeof(KERNELS[0]))

/* ---------------- helpers ---------------- */

static void init_vectors(double* A, double* B, int64_t n)
{
    /* Formula (document this in the README):
     *     A[i] = sin(i * 0.001)
     *     B[i] = cos(i * 0.002)
     */
    for (int64_t i = 0; i < n; i++) {
        A[i] = sin((double)i * 0.001);
        B[i] = cos((double)i * 0.002);
    }
}

/* Sum of |A[i]*B[i]| -- scale for the tolerance check (not timed). */
static double abs_product_sum(const double* A, const double* B, int64_t n)
{
    double s = 0.0;
    for (int64_t i = 0; i < n; i++)
        s += fabs(A[i] * B[i]);
    return s;
}

/* Runs fn RUNS times, returns average time in milliseconds.
 * Only the kernel call sits between the two counter reads. */
static double time_kernel(dot_kernel_fn fn, int64_t n,
    const double* A, const double* B, double* Z)
{
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    fn(n, A, B, Z);   /* warm-up run, not timed (TODO: mention in README) */

    double total_ms = 0.0;
    for (int r = 0; r < RUNS; r++) {
        QueryPerformanceCounter(&t0);
        fn(n, A, B, Z);
        QueryPerformanceCounter(&t1);
        total_ms += (double)(t1.QuadPart - t0.QuadPart) * 1000.0
            / (double)freq.QuadPart;
    }
    return total_ms / RUNS;
}

/* SIMD kernels add in a different order than C, so results will not be
 * bit-identical. Pass if |z - ref| <= REL_TOL * sum|A[i]*B[i]|.
 * (Scaling by the absolute sum avoids false FAILs when ref is near 0.) */
static int check_result(double z, double ref, double scale)
{
    double tol = REL_TOL * (scale > 0.0 ? scale : 1.0);
    return fabs(z - ref) <= tol;
}

/* Allocates, initializes, runs all kernels for one n. Returns 0 on success. */
static int run_size(int64_t n, int do_timing)
{
    size_t bytes = (size_t)n * sizeof(double);
    double* A = (double*)_aligned_malloc(bytes, ALIGNMENT);
    double* B = (double*)_aligned_malloc(bytes, ALIGNMENT);
    if (!A || !B) {
        printf("n = %lld: allocation of 2 x %.2f GB failed\n",
            (long long)n, bytes / (1024.0 * 1024.0 * 1024.0));
        _aligned_free(A);
        _aligned_free(B);
        return -1;
    }

    init_vectors(A, B, n);
    double scale = abs_product_sum(A, B, n);

    double results[NUM_KERNELS];
    double times[NUM_KERNELS];

    for (size_t k = 0; k < NUM_KERNELS; k++) {
        if (do_timing) {
            times[k] = time_kernel(KERNELS[k].fn, n, A, B, &results[k]);
        }
        else {
            KERNELS[k].fn(n, A, B, &results[k]);
            times[k] = 0.0;
        }
    }

    double ref = results[0];
    for (size_t k = 0; k < NUM_KERNELS; k++) {
        const char* verdict = (k == 0) ? "REF"
            : (check_result(results[k], ref, scale) ? "PASS" : "FAIL");
        if (do_timing)
            printf("%12lld  %-11s  Z = %22.12f  avg = %10.4f ms  %s\n",
                (long long)n, KERNELS[k].name, results[k], times[k], verdict);
        else
            printf("%12lld  %-11s  Z = %22.12f  %s\n",
                (long long)n, KERNELS[k].name, results[k], verdict);
    }
    printf("\n");

    _aligned_free(A);
    _aligned_free(B);
    return 0;
}

/* ---------------- main ---------------- */

int main(void)
{
    printf("Vector dot product  |  build: %s  |  runs per kernel: %d\n\n",
        BUILD_NAME, RUNS);

    printf("=== Boundary / correctness tests (untimed) ===\n");
    for (size_t i = 0; i < sizeof(BOUNDARY_SIZES) / sizeof(BOUNDARY_SIZES[0]); i++)
        run_size(BOUNDARY_SIZES[i], 0);

    printf("=== Benchmarks (average of %d runs, kernel only) ===\n", RUNS);
    for (size_t i = 0; i < sizeof(BENCH_SIZES) / sizeof(BENCH_SIZES[0]); i++)
        run_size(BENCH_SIZES[i], 1);

    return 0;
}