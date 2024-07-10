set(LLVM_TARGET_DEFINITIONS ArithExtToArith.td)
mlir_tablegen(ArithExtToArith.h.inc -gen-pass-decls -name ArithExt)
mlir_tablegen(ArithExtToArith.cpp.inc -gen-rewriters -name ArithExt)
add_public_tablegen_target(MLIRArithExtTransformsIncGen)

add_mlir_dialect_library(MLIRArithExtTransforms
	ArithExtToArith.cpp
	
	ADDITIONAL_HEADER_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/IR

	DEPENDS
	MLIRArithExtTransformsIncGen
	
	LINK_LIBS PUBLIC
	MLIRArithExt
	MLIRIR
	MLIRInferTypeOpInterface
	MLIRArithDialect
	MLIRSupport
	MLIRDialect
	MLIRIR
)
