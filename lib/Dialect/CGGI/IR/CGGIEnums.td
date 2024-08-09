#ifndef LIB_DIALECT_CGGI_IR_CGGIENUMS_TD_
#define LIB_DIALECT_CGGI_IR_CGGIENUMS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"

// Need to define the gates as "bg_*" since strings "and", "or" are reserved
def BooleanGateEnumAttr : I32EnumAttr<"BoolGateEnum",
    "An enum attribute representing a boolean gate using u8 int",
    [
  I32EnumAttrCase<"bg_and", 0>,
  I32EnumAttrCase<"bg_nand", 1>,
    I32EnumAttrCase<"bg_or", 2>,
  I32EnumAttrCase<"bg_nor", 3>,
    I32EnumAttrCase<"bg_xor", 4>,
  I32EnumAttrCase<"bg_xnor", 5>
]> {
    let cppNamespace = "::mlir::heir::cggi";

}


#endif  // LIB_DIALECT_CGGI_IR_CGGIENUMS_TD_
