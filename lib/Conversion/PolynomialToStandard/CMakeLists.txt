set(LLVM_TARGET_DEFINITIONS PolynomialToStandard.td)
mlir_tablegen(PolynomialToStandard.h.inc -gen-pass-decls -name Polynomial)
add_public_tablegen_target(MLIRPolynomialTransformsIncGen)

add_mlir_dialect_library(MLIRPolynomialTransforms
	PolynomialToStandard.cpp
	
	ADDITIONAL_HEADER_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/IR

	DEPENDS
	MLIRPolynomialTransformsIncGen

	LINK_LIBS PUBLIC

	MLIRHEIRUtils
	
        MLIRAffineDialect
        MLIRArithDialect
	MLIRDialectUtils
	MLIRFuncDialect
	MLIRFuncTransforms
	MLIRLLVMDialect
	MLIRLinalgDialect
        MLIRSCFDialect
	MLIRTensorDialect
	
	MLIRPolynomialDialect

	MLIRTransformUtils
	MLIRPass
	MLIRInferTypeOpInterface
	MLIRArithDialect
	MLIRSupport
	MLIRDialect
	MLIRIR
)
