#ifndef HEIR_LIB_DIALECT_ARITHEXT_IR_ARITHTYPES_H_
#define HEIR_LIB_DIALECT_ARITHEXT_IR_ARITHTYPES_H_

include "lib/Dialect/ArithExt/IR/ArithExtDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class ArithExt_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ArithExt_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // HEIR_LIB_DIALECT_ARITHEXT_IR_ARITHTYPES_H_
