; dot_product_ymm.asm  --  AVX2 SIMD kernel using YMM (owner: Person 3)
; NASM syntax, Windows x64 ABI
;
; void dot_product_ymm(int64_t n, const double* A, const double* B, double* Z)
;   RCX = n, RDX = A, R8 = B, R9 = Z
;
; STUB provided by Person 1 so the project links. Person 3 replaces the body.
; Expected structure:
;   1. main loop: n / 4 iterations, 4 doubles per step (vmovupd / vmulpd / vaddpd on ymm)
;   2. horizontal reduction of the 4 lanes (vextractf128 + add, then add the 2 halves)
;   3. tail: n % 4 leftover elements (scalar)
;   4. store result to [r9]
;   5. vzeroupper before ret
; Only use YMM0-YMM5 unless you save/restore XMM6-XMM15.

bits 64
default rel

section .text
global dot_product_ymm

dot_product_ymm:
    vxorpd  xmm0, xmm0, xmm0
    vmovsd  [r9], xmm0          ; placeholder: always returns 0.0 (will FAIL the check)
    vzeroupper
    ret