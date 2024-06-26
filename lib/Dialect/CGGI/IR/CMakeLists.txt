add_mlir_dialect(CGGIOps cggi)

set(LLVM_TARGET_DEFINITIONS CGGIAttributes.td)
mlir_tablegen(CGGIAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect="cggi")
mlir_tablegen(CGGIAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect="cggi")
add_public_tablegen_target(MLIRCGGIAttributesIncGen)
