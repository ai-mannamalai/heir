add_subdirectory(IR)
add_mlir_dialect_library(MLIRJaxite
        IR/JaxiteDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRJaxiteOpsIncGen
    
	LINK_LIBS PUBLIC
    	MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
