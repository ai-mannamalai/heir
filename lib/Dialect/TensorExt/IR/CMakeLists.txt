add_mlir_dialect(TensorExtOps tensor_ext)


set(LLVM_TARGET_DEFINITIONS TensorExtAttributes.td)
mlir_tablegen(TensorExtAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect="tensor_ext")
mlir_tablegen(TensorExtAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect="tensor_ext")
add_public_tablegen_target(MLIRTensorExtAttributesIncGen)

add_dependencies(MLIRTensorExtOpsIncGen MLIRTensorExtAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS TensorExtCanonicalization.td)
mlir_tablegen(TensorExtOpsCanonicalization.cpp.inc -gen-rewriters -dialect="tensor_ext")
add_public_tablegen_target(MLIRTensorExtCanonicalizationIncGen)

add_dependencies(MLIRTensorExtOpsIncGen MLIRTensorExtCanonicalizationIncGen)
