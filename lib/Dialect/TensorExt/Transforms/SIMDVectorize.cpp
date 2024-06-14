#include "lib/Dialect/TensorExt/Transforms/SIMDVectorize.h"

#include <stdbool.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <utility>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_NAME "simd-vectorize"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_SIMDVECTORIZE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

namespace {

static bool isNonCiphertextDegreeTensorType(Type type, int64_t numSlots) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    assert(tensorTy.getRank() == 1 &&
           "SIMD packing pass only supports 1-D tensors");
    auto dimension = tensorTy.getShape()[0];
    return dimension != numSlots;
  }

  return false;
}

}  // namespace

class SecretTensorTypeConverter : public TypeConverter {
 public:
  SecretTensorTypeConverter(int numSlots) {
    addConversion([](Type type) { return type; });

    addConversion([&](RankedTensorType type) {
      type.dump();
      if (isNonCiphertextDegreeTensorType(type, numSlots_)) {
        std::cout << "CONVERTING NONCIPHERTEXT TENSOR TYPE\n";
        type.dump();
        auto ctx = type.getContext();
        auto dimension = type.getShape()[0];
        auto newShape = llvm::ArrayRef<int64_t>(this->numSlots_);
        // FIXME: assume smaller for now
        auto nextPowerOf2 = llvm::PowerOf2Ceil(dimension);
        auto padding = DenseI64ArrayAttr::get(
            ctx, llvm::ArrayRef<int64_t>(nextPowerOf2 - dimension));
        return RankedTensorType::get(
            newShape, type.getElementType(),
            SIMDPackingAttr::get(
                ctx, DenseI64ArrayAttr::get(ctx, type.getShape()), padding,
                DenseI64ArrayAttr::get(ctx, newShape), 0));
      }
      return type;
    });

    addConversion([&](secret::SecretType secretType) -> Type {
      auto convertedTensorType = this->convertType(secretType.getValueType());
      return secret::SecretType::get(convertedTensorType);
    });

    numSlots_ = numSlots;
  }
  int numSlots_;
};

struct ConvertAny : public ConversionPattern {
  ConvertAny(const TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {}

  // generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    op->dump();
    // Don't match on func, since there is a specialized version for that.
    if (isa<func::FuncOp>(op) || isa<secret::GenericOp>(op)) return failure();

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();

    SmallVector<std::unique_ptr<Region>, 1> regions;
    IRMapping mapping;
    for (auto &r : op->getRegions()) {
      Region *newRegion = new Region();
      rewriter.cloneRegionBefore(r, *newRegion, newRegion->end(), mapping);
      if (failed(rewriter.convertRegionTypes(newRegion, *this->typeConverter)))
        return failure();
      regions.emplace_back(newRegion);
    }

    Operation *newOp = rewriter.create(OperationState(
        op->getLoc(), op->getName().getStringRef(), operands, newResultTypes,
        op->getAttrs(), op->getSuccessors(), regions));

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertGenericOp : public OpConversionPattern<secret::GenericOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertGenericOp(MLIRContext *context)
      : OpConversionPattern<secret::GenericOp>(context) {}

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> newResultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(),
                                           newResultTypes))) {
      return mlir::failure();
    }

    auto newOp = rewriter.create<secret::GenericOp>(
        op->getLoc(), adaptor.getOperands(), newResultTypes,
        [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
          IRMapping mp;
          for (BlockArgument blockArg : op.getBody()->getArguments()) {
            mp.map(blockArg, blockArguments[blockArg.getArgNumber()]);
          }
          for (auto &op : op.getBody()->getOperations()) {
            b.clone(op, mp);
          }
        });
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct SIMDVectorize : impl::SIMDVectorizeBase<SIMDVectorize> {
  using SIMDVectorizeBase::SIMDVectorizeBase;

  void runOnOperation() override {
    // Promote secret tensor types of arbitrary dimension into one specified by
    // numSlots.
    // Open questions: should this only run on secret tensor types? For
    // plaintext matrices, we likely need to pack into a ciphertext if it is
    // cheaper than many MulPlains. (e.g. a dot product with a plaintext weight
    // matrix).
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    SecretTensorTypeConverter typeConverter(numSlots);
    target.addLegalOp<ModuleOp>();
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      op->dump();
      return typeConverter.isLegal(op);
    });

    /* Require these since markUnknownOpDynamicallyLegal doesn't seem to work.
    target.addDynamicallyLegalOp<secret::GenericOp>(
        [&](secret::GenericOp op) { return typeConverter.isLegal(op); });

    target.addDynamicallyLegalOp<secret::YieldOp>(
        [&](secret::YieldOp op) { return typeConverter.isLegal(op); });

    target.addDynamicallyLegalOp<arith::AddIOp>(
        [&](arith::AddIOp op) { return typeConverter.isLegal(op); });
    */

    // TODO: ensure that insert and extract indices are updated.
    patterns.add<ConvertGenericOp, ConvertAny>(typeConverter, context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
