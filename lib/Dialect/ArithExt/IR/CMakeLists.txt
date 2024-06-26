add_mlir_dialect(ArithExtOps arith_ext)

add_mlir_doc(ArithExtDialect ArithExtDialect arithext/ -gen-dialect-doc --dialect=arithext)
add_mlir_doc(ArithExtOps ArithExtOps arithext/ -gen-op-doc -gen-attrdef-doc)
