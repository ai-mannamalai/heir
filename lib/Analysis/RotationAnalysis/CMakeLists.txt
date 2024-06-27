add_mlir_dialect_library(MLIRHEIRAnalysisROT
        RotationAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

        DEPENDS
        MLIRTensorExtOpsIncGen

	LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)
