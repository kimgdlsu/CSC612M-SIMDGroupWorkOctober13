; dot_product_asm.asm  --  scalar (non-SIMD) x86-64 kernel (owner: Person 1)
; NASM syntax, Windows x64 ABI
;
; void dot_product_asm(int64_t n, const double* A, const double* B, double* Z)
;   RCX = n, RDX = A, R8 = B, R9 = Z
;
; Scalar double instructions only (movsd / mulsd / addsd):
; one element per iteration, same order as the C version.

bits 64
default rel

section .text
global dot_product_asm

dot_product_asm:
    xorpd   xmm0, xmm0          ; sum = 0.0

    test    rcx, rcx            ; n <= 0 ?  -> nothing to add
    jle     .done

.loop:
    movsd   xmm1, [rdx]         ; xmm1 = A[i]
    mulsd   xmm1, [r8]          ; xmm1 = A[i] * B[i]
    addsd   xmm0, xmm1          ; sum += A[i] * B[i]

    add     rdx, 8              ; next A element (8 bytes per double)
    add     r8, 8               ; next B element
    dec     rcx                 ; one fewer element left
    jnz     .loop

.done:
    movsd   [r9], xmm0          ; *Z = sum
    ret