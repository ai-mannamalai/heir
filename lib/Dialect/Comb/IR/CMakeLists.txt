add_mlir_dialect(CombOps comb)
add_mlir_doc(CombDialect CombDialect comb/ -gen-dialect-doc --dialect=comb)
add_mlir_doc(CombOps CombOps comb/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS CombDialect.td)
mlir_tablegen(CombEnums.h.inc -gen-enum-decls -dialect=comb)
mlir_tablegen(CombEnums.cpp.inc -gen-enum-defs -dialect=comb)
add_public_tablegen_target(MLIRCombDialectEnumsIncGen)

add_dependencies(MLIRCombOpsIncGen MLIRCombDialectEnumsIncGen)
