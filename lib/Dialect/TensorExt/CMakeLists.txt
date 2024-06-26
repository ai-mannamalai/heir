add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRTensorExt
        IR/TensorExtDialect.cpp
        Transforms/CollapseInsertionChains.cpp
        Transforms/InsertRotate.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_S}/IR

        DEPENDS
        MLIRTensorExtOpsIncGen
        MLIRTensorExtAttributesIncGen
        MLIRTensorExtPassesIncGen
        
	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
