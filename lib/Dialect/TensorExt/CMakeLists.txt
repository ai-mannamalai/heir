add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRTensorExt
        IR/TensorExtDialect.cpp
        Transforms/CollapseInsertionChains.cpp
        Transforms/InsertRotate.cpp
        Transforms/RotateAndReduce.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRTensorExtOpsIncGen
        MLIRTensorExtPassesIncGen
        MLIRTensorExtInsertRotateIncGen
        MLIRTensorExtCanonicalizationIncGen
        
	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
