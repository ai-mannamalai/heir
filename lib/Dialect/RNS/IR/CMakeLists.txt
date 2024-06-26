add_mlir_dialect(RNSOps rns)

set(LLVM_TARGET_DEFINITIONS RNSTypeInterfaces.td)
mlir_tablegen(RNSTypeInterfaces.h.inc  --gen-type-interface-decls  -name rns)
mlir_tablegen(RNSTypeInterfaces.cpp.inc  --gen-type-interface-defs -name rns)

add_public_tablegen_target(MLIRRNSTypeInterfaces)
add_dependencies(MLIRRNSOpsIncGen MLIRRNSTypeInterfaces)
