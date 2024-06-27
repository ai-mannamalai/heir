
add_mlir_dialect_library(MLIRVerilogtTarget
        VerilogEmitter.cpp
        
        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRSecretOpsIncGen

	LINK_LIBS PUBLIC
	MLIRSecret
    MLIRIR
    MLIRInferTypeOpInterface
)
