add_subdirectory(ArithExt)
add_subdirectory(Jaxite)
add_subdirectory(PolyExt)
add_subdirectory(LWE)
add_subdirectory(Openfhe)
add_subdirectory(RNS)
add_subdirectory(TfheRust)
add_subdirectory(TfheRustBool)

add_subdirectory(BGV)
add_subdirectory(CGGI)
# IR + Transforms
# Secret
# TensorExt

set(LLVM_TARGET_DEFINITIONS HEIRInterfaces.td)
mlir_tablegen(HEIRInterfaces.h.inc  --gen-op-interface-decls  -name heir)
mlir_tablegen(HEIRInterfaces.cpp.inc  --gen-op-interface-defs -name heir)
add_public_tablegen_target(MLIRHEIRInterfaces)

add_subdirectory(Comb) #- requires HEIR interface
add_dependencies(MLIRCombOpsIncGen MLIRHEIRInterfaces)
