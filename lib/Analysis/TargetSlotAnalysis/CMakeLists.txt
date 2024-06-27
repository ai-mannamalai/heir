add_mlir_dialect_library(MLIRHEIRAnalysisTSN
        TargetSlotAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

        DEPENDS
        MLIROpenfheOpsIncGen

	LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)
