#include "nvbit_tool.h"
#include "nvbit.h"

void nvbit_at_init() {
    printf("NVBit tool initialized!\n");
}

void nvbit_at_ctx_init(CUcontext ctx) {
    printf("Context initialized!\n");
}

void nvbit_at_function_load(CUcontext ctx, const CUfunction func) {
    const char* funcName = nvbit_get_func_name(ctx, func);
    printf("Instrumenting function: %s\n", funcName);

    // Example: inject a call before every instruction
    std::vector<Instr*> instrs;
    nvbit_get_instrs(ctx, func, instrs);
    for (auto i : instrs) {
        nvbit_insert_call(i, "my_callback", IPOINT_BEFORE);
    }
}
