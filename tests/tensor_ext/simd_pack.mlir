// RUN: heir-opt --secretize=entry-function=box_blur --wrap-generic --canonicalize --cse \
// RUN:   --simd-vectorize %s | FileCheck %s

module  {
  // CHECK-LABEL: @add
  func.func @add(%arg0: tensor<16xi16>, %arg1: tensor<16xi16>) -> tensor<16xi16> {
    %0 = arith.addi %arg0, %arg1 : tensor<16xi16>
    return %0 : tensor<16xi16>
  }
}