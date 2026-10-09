; dot_product_ymm.asm -- AVX2 SIMD kernel using YMM
; NASM syntax, Windows x64 ABI
;
; Parameters passed in:
;   RCX = n (number of elements)
;   RDX = pointer to Array A
;   R8  = pointer to Array B
;   R9  = pointer to Z (result variable)

bits 64
default rel

section .text
global dot_product_ymm

dot_product_ymm:
    ; Zero out ymm0 where we'll accumulate partial sums
    vxorpd      ymm0, ymm0, ymm0

    ; Make sure n > 0, otherwise skip everything
    test        rcx, rcx
    jle         .store_result

    ; Calculate loop counts
    mov         rax, rcx
    shr         rax, 2              ; rax = n / 4 (number of 4-element YMM chunks)
    mov         r10, rcx
    and         r10, 3              ; r10 = n % 4 (leftover elements)

    ; If n < 4, skip YMM loop and go straight to tail
    test        rax, rax
    jz          .tail_loop

.ymm_loop:
    ; Load 4 doubles from A and B
    vmovupd     ymm1, [rdx]
    vmovupd     ymm2, [r8]

    ; Multiply elements: ymm1 = A[i..i+3] * B[i..i+3]
    vmulpd      ymm1, ymm1, ymm2

    ; Add products to running totals in ymm0
    vaddpd      ymm0, ymm0, ymm1

    ; Move array pointers by 32 bytes (4 doubles * 8 bytes)
    add         rdx, 32
    add         r8, 32

    dec         rax
    jnz         .ymm_loop

    ; Reduce 4 YMM lanes down to 1 scalar double
    ; Extract upper 2 doubles (lanes 2 & 3) into xmm1
    vextractf128 xmm1, ymm0, 1

    ; Add upper 2 doubles to lower 2 doubles
    vaddpd      xmm0, xmm0, xmm1

    ; Add remaining 2 doubles together horizontally
    vhaddpd     xmm0, xmm0, xmm0

.tail_loop:
    ; Check for leftover elements (n % 4)
    test        r10, r10
    jz          .store_result

.tail_step:
    ; Process 1 leftover double scalar-style
    vmovsd      xmm1, [rdx]
    vmulsd      xmm1, xmm1, [r8]
    vaddsd      xmm0, xmm0, xmm1

    ; Advance pointers by 8 bytes (1 double)
    add         rdx, 8
    add         r8, 8
    dec         r10
    jnz         .tail_step

.store_result:
    ; Save final double result to *Z
    vmovsd      [r9], xmm0

    ; Reset upper YMM register state before returning
    vzeroupper
    ret