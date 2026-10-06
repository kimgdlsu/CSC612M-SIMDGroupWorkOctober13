; dot_product_xmm.asm  --  AVX2 SIMD kernel using XMM (owner: Person 2)
; NASM syntax, Windows x64 ABI
;
; void dot_product_xmm(int64_t n, const double* A, const double* B, double* Z)
;   RCX = n, RDX = A, R8 = B, R9 = Z
;
; STUB provided by Person 1 so the project links. Person 2 replaces the body.
; Expected structure:
;   1. main loop: n / 2 iterations, 2 doubles per step (vmovupd / vmulpd / vaddpd on xmm)
;   2. horizontal reduction of the 2 lanes -> one scalar
;   3. tail: 1 leftover element when n is odd (scalar)
;   4. store result to [r9]
; Use vmovupd (unaligned) unless the group agrees all buffers are aligned.
; Only use XMM0-XMM5 unless you save/restore XMM6-XMM15.

bits 64
default rel

section .text
global dot_product_xmm

dot_product_xmm:
    vxorpd  xmm0, xmm0, xmm0
    vmovsd  [r9], xmm0          ; placeholder: always returns 0.0 (will FAIL the check)
    ret