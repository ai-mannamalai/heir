add_subdirectory(IR)

add_mlir_dialect_library(MLIROpenfhe
        IR/OpenfheDialect.cpp
        
        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIROpenfheOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
)
add_dependencies(MLIROpenfhe MLIRLWE)