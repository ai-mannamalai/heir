#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"

#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOpsDialect.cpp.inc"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOpsTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.cpp.inc"

namespace mlir {
namespace heir {
namespace tfhe_rust_bool {

void TfheRustBoolDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.cpp.inc"
      >();
}

}  // namespace tfhe_rust_bool
}  // namespace heir
}  // namespace mlir
