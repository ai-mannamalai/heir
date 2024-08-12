set(LLVM_TARGET_DEFINITIONS TfheRustBoolDialect.td)
mlir_tablegen(TfheRustBoolDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TfheRustBoolDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRTfheRustBoolIncGen)

set(LLVM_TARGET_DEFINITIONS TfheRustBoolOps.td)
mlir_tablegen(TfheRustBoolOps.h.inc -gen-op-decls)
mlir_tablegen(TfheRustBoolOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTfheRustBoolOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TfheRustBoolTypes.td)
mlir_tablegen(TfheRustBoolTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TfheRustBoolTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRTfheRustBoolOpsTypesIncGen)

set(LLVM_TARGET_DEFINITIONS TfheRustBoolAttributes.td)
mlir_tablegen(TfheRustBoolAttributes.cpp.inc -gen-attrdef-defs -typedefs-dialect=tfhe_rust_boo)
mlir_tablegen(TfheRustBoolAttributes.h.inc -gen-attrdef-decls -typedefs-dialect=tfhe_rust_bool)
add_public_tablegen_target(MLIRTfheRustBoolAttributesIncGen)
