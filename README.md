# CSC612M SIMD Vector Dot Product

**Author:** Khladyn  
**Course:** CSC612M - Advanced Computer Architecture  
**Topic:** Scalar C, x86-64 assembly, XMM SIMD, and YMM SIMD dot-product kernels

## Project Overview

This project computes the dot product of two double-precision vectors:

$$Z = \sum_{i=0}^{n-1} A_i B_i$$

It compares four implementations using a shared C driver. The C result is the reference for correctness checks.

| Kernel | Implementation |
| --- | --- |
| C | Scalar loop in `dot_product_c.c` |
| x86-64 ASM | Scalar floating-point instructions in `dot_product_asm.asm` |
| SIMD XMM | Two doubles per 128-bit operation in `dot_product_xmm.asm` |
| SIMD YMM | Four doubles per 256-bit operation in `dot_product_ymm.asm` |

The assembly uses NASM syntax and the Windows x64 calling convention. The common project interface is declared in `kernels.h`: `n`, `A`, `B`, and an output pointer `Z` are passed in that order. On Windows x64 these arrive in `RCX`, `RDX`, `R8`, and `R9` respectively.

## Input Initialization

The driver initializes the vectors deterministically:

```text
A[i] = sin(i * 0.001)
B[i] = cos(i * 0.002)
```

This makes runs repeatable and provides nontrivial floating-point data.

## Building and Running

This is a Windows x64 Visual Studio project; `main.c` uses Windows timing and aligned-allocation APIs. A native x86-64 Windows environment is recommended for AVX/YMM validation and performance measurements.

Prerequisites:

- Visual Studio with the Desktop development with C++ workload and the project toolset (`v145`).
- NASM installed and available as `nasm` on `PATH`.
- An x86-64 processor/runner with AVX support to execute the XMM/YMM kernels.

Open `ConsoleApplication1.slnx`, choose the `x64` platform and either `Debug` or `Release`, then build and run the console application. The project assembles each `.asm` file with `nasm -fwin64` and writes its object file into the Visual Studio intermediate directory.

The GitHub Actions workflow in `.github/workflows/build.yml` builds and runs the `Debug|x64` configuration on `windows-latest`. It does not currently build Release, upload screenshots, or fail the job based on printed `FAIL` lines; inspect the program output when using CI as validation.

## Kernel Notes

- **C:** Accumulates `A[i] * B[i]` in a scalar `double` and stores the result through `Z`.
- **Scalar assembly:** Uses `movsd`, `mulsd`, and `addsd` to process one product per iteration.
- **XMM:** Processes pairs with packed double operations, reduces the two accumulated lanes, then handles an odd leftover element.
- **YMM:** Processes groups of four with packed double operations, reduces the four partial sums, then handles `n % 4` leftover elements. The assembly uses 256-bit AVX YMM operations; the assignment describes this kernel as AVX2/YMM.

The XMM and YMM reductions change the order of floating-point additions. Their results can therefore differ slightly from the sequential C result without indicating an error.

## Correctness and Boundary Tests

The driver compares each non-C result against the C reference using:

```text
abs(result - reference) <= 1e-9 * sum(abs(A[i] * B[i]))
```

The scale avoids an overly strict absolute tolerance when the reference is near zero. The driver currently runs untimed tests for `n = 1, 2, 3, 5, 7, 1003, 1048577, 1048579`. These exercise all modulo-four remainders, including sizes smaller than one YMM vector.

The assignment's Person 3 checklist specifically asks for `n = 1000, 1001, 1002, 1003, 1004`. Only `1003` is in the current shared test list; coordinate with Person 1 to add the other requested sizes before claiming that exact checklist is complete.

### Correctness Output

The following results were supplied with the project report. They show all four kernels passing the listed cases; rerun the current integrated build before treating this excerpt as current verification evidence.

```text
n         C reference             Scalar ASM             XMM                    YMM
1         0.000000000000           PASS                   PASS                   PASS
2         0.000999997833           PASS                   PASS                   PASS
3         0.002999980500           PASS                   PASS                   PASS
5         0.009999783335           PASS                   PASS                   PASS
7         0.020999044512           PASS                   PASS                   PASS
1003      100.935634783182         PASS                   PASS                   PASS
1048577   134.781920612681         PASS                   PASS                   PASS
1048579   134.596594493716         PASS                   PASS                   PASS
```

### Screenshots

Add the required screenshots after capturing them from the final integrated executable:

- `[TODO: insert program output and timing screenshot for all benchmark cases]`
- `[TODO: insert C correctness screenshot]`
- `[TODO: insert scalar x86-64 assembly correctness screenshot]`
- `[TODO: insert XMM correctness and boundary screenshots]`
- `[TODO: insert YMM correctness and boundary screenshots]`

## Timing Methodology

The driver benchmarks `n = 2^20`, `2^26`, and `2^28`. Each kernel receives one untimed warm-up call, followed by 30 timed calls. Only the kernel function call is between the performance-counter reads; vector initialization, allocation, and correctness-scale calculation are outside the timed region. The displayed time is the average of the 30 timed calls.

The source contains the benchmark sizes and methodology, but the measurements below do not identify the build configuration, processor, memory configuration, or compiler settings. Record those details alongside the final Debug and Release measurements; do not assume this single table satisfies both-build reporting.

At `n = 2^28`, each vector occupies 2 GiB, for 4 GiB total input data. The largest size may need to be reduced on machines with insufficient available memory.

## Supplied Benchmark Results

The following measurements and outputs were provided for this report. Speedup is calculated as `C average time / kernel average time`. Build mode and test-machine specifications were not included with the supplied values.

| Vector size | Elements | Kernel | Output Z | Average (ms) | Speedup vs. C |
| --- | ---: | --- | ---: | ---: | ---: |
| $2^{20}$ | 1,048,576 | C | 134.872794645111 | 4.1144 | 1.00x |
| $2^{20}$ | 1,048,576 | x86-64 ASM | 134.872794645111 | 1.0252 | 4.01x |
| $2^{20}$ | 1,048,576 | SIMD XMM | 134.872794645113 | 0.5551 | 7.41x |
| $2^{20}$ | 1,048,576 | SIMD YMM | 134.872794645121 | 0.3645 | 11.29x |
| $2^{26}$ | 67,108,864 | C | -585.732854649973 | 256.8256 | 1.00x |
| $2^{26}$ | 67,108,864 | x86-64 ASM | -585.732854649973 | 69.9938 | 3.67x |
| $2^{26}$ | 67,108,864 | SIMD XMM | -585.732854650150 | 43.1829 | 5.95x |
| $2^{26}$ | 67,108,864 | SIMD YMM | -585.732854650142 | 38.3548 | 6.70x |
| $2^{28}$ | 268,435,456 | C | 72.826491389550 | 1030.1068 | 1.00x |
| $2^{28}$ | 268,435,456 | x86-64 ASM | 72.826491389550 | 279.2728 | 3.69x |
| $2^{28}$ | 268,435,456 | SIMD XMM | 72.826491389638 | 178.6872 | 5.76x |
| $2^{28}$ | 268,435,456 | SIMD YMM | 72.826491389353 | 160.1395 | 6.43x |

### Performance Discussion

In the supplied measurements, YMM is the fastest kernel at every listed size. At `n = 2^28`, its average is about 6.43 times faster than C and about 1.12 times faster than XMM. Although YMM processes twice as many doubles per packed operation as XMM, it is not twice as fast: wider arithmetic does not guarantee a proportional end-to-end speedup.

For large vectors, the kernel must read 4 GiB of input data per call at `n = 2^28`. Once data movement becomes a bottleneck, memory bandwidth and cache behavior can limit the benefit of additional SIMD width. Smaller inputs may fit more readily in cache and show different relative performance. These measurements alone do not establish the bottleneck; hardware counters or controlled cache/bandwidth experiments would be needed to confirm it.

SIMD addition is performed in a different order from the scalar C loop, so small differences in the final decimal places are expected under IEEE 754 floating-point arithmetic. The provided results are within the driver's stated tolerance.

The reported scalar-assembly and SIMD speedups depend on build configuration and compiler optimization. Since the supplied table does not say whether it came from Debug or Release, avoid attributing the gains solely to reduced loop overhead until both configurations and their generated code have been compared.

## Problems and Methodology Notes

- The development machine is an Apple Silicon Mac. It cannot execute x86-64 AVX/YMM instructions natively, so local timings on the Mac are not suitable for this x86-64 kernel comparison.
- The project uses Windows APIs and the Windows x64 calling convention. The provided GitHub Actions workflow is the available x64 Windows build/run path, but results should include the runner and build configuration for reproducibility.
- The Visual Studio project previously used a machine-specific NASM path. It now resolves `nasm` through `PATH`, allowing the local build and workflow to use an installed NASM executable.
- The assignment asks for both Debug and Release results. The current workflow covers Debug only; run and report Release separately.
- The YMM kernel's packed double instructions use the AVX instruction subset for 256-bit floating-point vectors; those particular operations are not AVX2-only. Confirm the course rubric accepts this instruction selection for its AVX2/YMM requirement.

## Remaining Report Items

- Add the requested screenshots from the final integrated run.
- Add explicit `1000` through `1004` boundary tests in coordination with the shared-driver owner, then capture their results.
- Record separate Debug and Release timing tables and the CPU/compiler environment.
- Confirm whether the YMM instruction selection meets the instructor's AVX2 requirement.