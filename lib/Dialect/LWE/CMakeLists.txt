add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRLWE
        IR/LWEDialect.cpp
        Transforms/SetDefaultParameters.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_S}/IR

        DEPENDS
        MLIRLWEOpsIncGen
        MLIRLWEAttributesIncGen
        MLIRLWEPassesIncGen
        
	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
