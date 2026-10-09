; dot_product_ymm.asm -- AVX2 SIMD kernel using YMM
; NASM syntax, Windows x64 ABI
;
; void dot_product_ymm(int64_t n, const double* A, const double* B, double* Z)
;   RCX = n, RDX = A, R8 = B, R9 = Z

bits 64
default rel

section .text
global dot_product_ymm

dot_product_ymm:
    ; Zero out ymm0 (will hold the 4 partial vector sums)
    vxorpd      ymm0, ymm0, ymm0

    ; Check if n <= 0
    test        rcx, rcx
    jle         .store_result

    ; Calculate main loop iterations: rax = n / 4, r10 = n % 4
    mov         rax, rcx
    shr         rax, 2              ; rax = n / 4
    mov         r10, rcx
    and         r10, 3              ; r10 = n % 4 (remainder)

    test        rax, rax
    jz          .tail_loop

.ymm_loop:
    ; Load 4 doubles from A and B (unaligned load)
    vmovupd     ymm1, [rdx]
    vmovupd     ymm2, [r8]

    ; Multiply A[i..i+3] * B[i..i+3]
    vmulpd      ymm1, ymm1, ymm2

    ; Accumulate into ymm0
    vaddpd      ymm0, ymm0, ymm1

    ; Advance pointers by 32 bytes (4 doubles * 8 bytes)
    add         rdx, 32
    add         r8, 32

    dec         rax
    jnz         .ymm_loop

    ; -------------------------------------------------------------
    ; Horizontal reduction of ymm0 (4 lanes -> 1 double in xmm0)
    ; -------------------------------------------------------------
    ; Extract upper 128 bits (lanes 2 & 3) of ymm0 into xmm1
    vextractf128 xmm1, ymm0, 1

    ; Add upper 128 bits (xmm1) to lower 128 bits (xmm0)
    ; xmm0 now contains [ (lane 0 + lane 2), (lane 1 + lane 3) ]
    vaddpd      xmm0, xmm0, xmm1

    ; Horizontal add adjacent pairs in xmm0
    ; xmm0 lower double = (lane 0 + lane 2) + (lane 1 + lane 3)
    vhaddpd     xmm0, xmm0, xmm0

.tail_loop:
    test        r10, r10
    jz          .store_result

.tail_step:
    ; Process leftover scalar elements (n % 4)
    vmovsd      xmm1, [rdx]
    vmulsd      xmm1, xmm1, [r8]
    vaddsd      xmm0, xmm0, xmm1

    add         rdx, 8
    add         r8, 8
    dec         r10
    jnz         .tail_step

.store_result:
    ; Store double result into *Z
    vmovsd      [r9], xmm0

    ; Clear upper YMM state before returning to avoid AVX-SSE transition penalty
    vzeroupper
    ret