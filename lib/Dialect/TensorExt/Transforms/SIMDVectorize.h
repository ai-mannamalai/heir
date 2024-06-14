#ifndef HEIR_LIB_DIALECT_TENSOREXT_TRANSFORMS_SIMDVECTORIZE_H_
#define HEIR_LIB_DIALECT_TENSOREXT_TRANSFORMS_SIMDVECTORIZE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DECL_SIMDVECTORIZE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // HEIR_LIB_DIALECT_TENSOREXT_TRANSFORMS_SIMDVECTORIZE_H_
