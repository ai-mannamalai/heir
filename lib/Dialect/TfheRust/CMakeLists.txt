add_subdirectory(IR)
add_mlir_dialect_library(MLIRTfheRust
        IR/TfheRustDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRTfheRustOpsIncGen
        
	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)

add_dependencies(MLIRLWE MLIRLWEAttributesIncGen)
