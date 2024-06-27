
add_mlir_dialect_library(MLIRJaxiteTarget
        JaxiteEmitter.cpp
        
        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRJaxiteOpsIncGen

	LINK_LIBS PUBLIC
	MLIRJaxite
    MLIRArithDialect
    MLIRAffineDialect
    MLIRIR
    MLIRInferTypeOpInterface
)
