#include "ppl/common/retcode.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

void RegisterParsers();

RetCode RegisterResourcesOnce() {
    static bool st_registered = false;
    if (!st_registered) {
        RegisterParsers();
        st_registered = true;
    }
    return RC_SUCCESS;
}

}}}
