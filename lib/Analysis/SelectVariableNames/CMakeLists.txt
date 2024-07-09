add_mlir_dialect_library(MLIRHEIRAnalysisSVN
        SelectVariableNames.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

	LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)
