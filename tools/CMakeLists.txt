get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

function(make_heir_exec  TARGET_NAME SOURCES)
        add_llvm_executable(${TARGET_NAME} PARTIAL_SOURCES_INTENDED ${SOURCES})
        llvm_update_compile_flags(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE 
        ${dialect_libs} ${conversion_libs}
        MLIROptLib MLIROpenfhe MLIRLWE MLIROpenfheTarget MLIRHEIRTargetUtils MLIRHEIRAnalysisSVN)
endfunction()

function(make_heirtx_exec  TARGET_NAME SOURCES)
        add_llvm_executable(${TARGET_NAME} PARTIAL_SOURCES_INTENDED ${SOURCES})
        llvm_update_compile_flags(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE 
        ${dialect_libs} ${conversion_libs}
        MLIROptLib MLIROpenfhe MLIRLWE MLIROpenfheTarget MLIRSecret MLIRVerilogTarget MLIRHEIRTargetUtils 
        MLIRTfheRust MLIRTfheRustBool MLIRHEIRAnalysisSVN
        MLIRHeirMemrefToArith
        MLIRHEIRTargetUtils
        )
endfunction()

set(OPENFHETARGET ${HEIRLIBDIR}/lib/Target/OpenFhePke/)
message("Target to be searched at " ${OPENFHETARGET})
set(cornami_translate_files ${openfhe_target_translation_files} ${CMAKE_CURRENT_SOURCE_DIR})
message("Found files for Cornami Translate: " ${openfhe_target_translation_files})
make_heir_exec(cornami-translate cornami-translate.cpp)
make_heirtx_exec(heir-translate heir-translate.cpp)
