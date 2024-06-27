
add_mlir_dialect_library(MLIRMetadataTarget
        MetadataEmitter.cpp
        
        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRLWEOpsIncGen

	LINK_LIBS PUBLIC
	MLIRArithDialect
    MLIRAffineDialect
    MLIRIR
    MLIRInferTypeOpInterface
)
