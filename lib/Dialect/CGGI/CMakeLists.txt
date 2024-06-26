add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRCGGI
        IR/CGGIDialect.cpp
        Transforms/SetDefaultParameters.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRCGGIOpsIncGen
        MLIRCGGIPassesIncGen
        MLIRCGGIAttributesIncGen
                
	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
add_dependencies(MLIRCGGI MLIRCGGIAttributesIncGen)
