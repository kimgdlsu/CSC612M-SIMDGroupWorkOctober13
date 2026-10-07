; dot_product_xmm.asm  --  AVX2 SIMD kernel using XMM (owner: Person 2)
; NASM syntax, Windows x64 ABI
;
; void dot_product_xmm(int64_t n, const double* A, const double* B, double* Z)
;   RCX = n, RDX = A, R8 = B, R9 = Z
;
; 128-bit XMM holds two doubles, so each packed step covers A[i]*B[i]
; and A[i+1]*B[i+1]. An odd n leaves one element for a scalar tail.
; The two accumulator lanes are then added into one double and stored
; through Z.
;
; vmovupd does not require alignment. main.c aligns to 32 bytes, but
; this kernel does not depend on that.
; Registers used: RAX, RCX, RDX, R8, R9, XMM0-XMM2. All are volatile.

bits 64
default rel

section .text
global dot_product_xmm

dot_product_xmm:
    vxorpd  xmm0, xmm0, xmm0 ; set everything to zero

    test rcx, rcx ; when it reaches 0, end the loop and save the Z
    jle .save

    mov rax, rcx ; write the n
    shr rax, 1 ; divide n by 2 since xmm can handle 2 at a time
    jz .odd ; If odd n

.pairs: ; While there are pairs to do
    vmovupd xmm1, [rdx] ; xmm1 = A[i+1], A[i]
    vmovupd xmm2, [r8]  ; xmm2 = B[i+1], B[i]
    vmulpd  xmm1, xmm1, xmm2 ; A[i+1]*B[i+1], A[i]*B[i]
    vaddpd  xmm0, xmm0, xmm1 ; accumulator

    add rdx, 16 ; advance them by two doubles
    add r8, 16
    dec rax
    jnz .pairs

.odd:
    test rcx, 1 ; check if last bit is zero meaning even, so skip
    jz .combine 

    vmovsd xmm1, [rdx]      ; A[n-1]
    vmulsd xmm1, xmm1, [r8] ; A[n-1] * B[n-1]
    vaddsd xmm0, xmm0, xmm1 ; add into the bottom half

.combine:
    vunpckhpd xmm1, xmm0, xmm0 ; Get the high half
    vaddsd    xmm0, xmm0, xmm1 ; Add the low and high half to the bottom

.save:
    vmovsd [r9], xmm0 ; Save final output
    vzeroupper ; clear for the YMMs
    ret
