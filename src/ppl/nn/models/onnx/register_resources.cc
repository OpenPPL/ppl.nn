#include "ppl/common/retcode.h"
using namespace ppl::common;

#include <mutex>

namespace ppl { namespace nn { namespace onnx {

void RegisterParsers();

RetCode RegisterResourcesOnce() {
    static std::once_flag st_registered;
    std::call_once(st_registered, []() {
        RegisterParsers();
    });
    return RC_SUCCESS;
}

}}}
