// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics

// Mixed fp8 operand types are not supported by nvgpu.mma.sync.
func.func @matmul_16x8x32_mixed_f8(
    %A: memref<16x32xf8E4M3FN>, %B: memref<32x8xf8E5M2>, %C: memref<16x8xf32>) {
  // expected-note @below {{target op}}
  linalg.matmul ins(%A, %B: memref<16x32xf8E4M3FN>, memref<32x8xf8E5M2>)
            outs(%C: memref<16x8xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{unsupported target op}}
    transform.nvgpu.rewrite_matmul_as_mma_sync %matmul
      : (!transform.any_op) -> ()
    transform.yield
  }
}
