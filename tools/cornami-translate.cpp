#include "lib/Target/Metadata/MetadataEmitter.h"
#include "lib/Target/OpenFhePke/OpenFhePkeEmitter.h"
#include "lib/Target/OpenFhePke/OpenFhePkeHeaderEmitter.h"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"  // from @llvm-project

int main(int argc, char **argv) {
  // OpenFHE
  mlir::heir::openfhe::registerToOpenFhePkeTranslation();
  mlir::heir::openfhe::registerToOpenFhePkeHeaderTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "HEIR Translation Tool"));
}
