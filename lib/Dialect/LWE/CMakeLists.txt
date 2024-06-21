add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRLWE
        IR/LWEDialect.cpp
        #IR/LWEAttributes.cpp
        Transforms/SetDefaultParameters.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_S}/IR

        DEPENDS
        MLIRLWEOpsIncGen
        MLIRLWEAttributesIncGen
        MLIRLWEPassesIncGen
        
	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)


set(LLVM_TARGET_DEFINITIONS PolynomialCanonicalization.td)
mlir_tablegen(PolynomialCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRPolynomialCanonicalizationIncGen)
add_dependencies(MLIRLWE MLIRLWEAttributesIncGen)
