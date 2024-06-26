add_subdirectory(IR)
add_mlir_dialect_library(MLIRRNS
        IR/RNSDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRRNSOpsIncGen
    
	LINK_LIBS PUBLIC
    	MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
