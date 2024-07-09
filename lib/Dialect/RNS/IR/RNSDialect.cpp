#include "lib/Dialect/RNS/IR/RNSDialect.h"

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define RNSOps
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"

// Generated definitions
#include "lib/Dialect/RNS/IR/RNSOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/RNS/IR/RNSOpsTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/RNS/IR/RNSOps.cpp.inc"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.cpp.inc"

namespace mlir {
namespace heir {
namespace rns {

void RNSDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/RNS/IR/RNSOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/RNS/IR/RNSOps.cpp.inc"
      >();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
