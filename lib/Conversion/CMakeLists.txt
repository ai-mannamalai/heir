add_mlir_dialect_library(MLIRHEIRUtils
	utils.cpp

	ADDITIONAL_HEADER_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/IR

	DEPENDS
	MLIRTensorExtOpsIncGen
	MLIRTensorExtPassesIncGen
	MLIRTensorExtInsertRotateIncGen
	MLIRTensorExtCanonicalizationIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	MLIRInferTypeOpInterface
	MLIRArithDialect
	MLIRSupport
	MLIRDialect
	MLIRIR
)

add_subdirectory(SecretToBGV)
add_subdirectory(ArithExtToArith)
add_subdirectory(PolynomialToStandard)
add_subdirectory(CombToCGGI)
add_subdirectory(BGVToOpenfhe)
add_subdirectory(BGVToPolynomial)
add_subdirectory(CGGIToTfheRust)
