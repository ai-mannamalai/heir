add_mlir_dialect(LWEOps lwe)

set(LLVM_TARGET_DEFINITIONS LWEAttributes.td)
mlir_tablegen(LWEAttributes -gen-typedef-decls -gen-attrdef-decls -typedefs-dialect=lwe)
mlir_tablegen(LWEAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=lwe)
mlir_tablegen(LWEAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=lwe)
add_public_tablegen_target(MLIRLWEAttributesIncGen)

add_mlir_doc(LWEDialect LWEDialect lwe/ -gen-dialect-doc --dialect=lwe)
add_mlir_doc(LWEOps LWEOps lwe/ -gen-op-doc)
add_mlir_doc(LWEAttributes LWEAttributes lwe/ -gen-attrdef-doc)
