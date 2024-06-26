add_subdirectory(IR)
add_mlir_dialect_library(MLIRArithExt
        IR/ArithExtDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRArithExtOpsIncGen
        
	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
