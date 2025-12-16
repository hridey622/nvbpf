/*
 * NV-BPF: eBPF-style Wrapper for NVBit
 * Loader/Backend - Auto-injection System
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <cstring>
#include "nvbit.h"
#include "nvbpf_types.h"
#include "nvbpf_hooks.h"

/* ============================================
 * Section 1: Instruction Classification
 * ============================================ */

namespace nvbpf {

/**
 * Check if instruction is a branch/jump
 */
inline bool is_branch_instruction(Instr* instr) {
    const char* opcode = instr->getOpcodeShort();
    // Common branch opcodes in SASS
    return (strncmp(opcode, "BRA", 3) == 0 ||
            strncmp(opcode, "JMP", 3) == 0 ||
            strncmp(opcode, "JMX", 3) == 0 ||
            strncmp(opcode, "BRX", 3) == 0 ||
            strncmp(opcode, "CALL", 4) == 0 ||
            strncmp(opcode, "RET", 3) == 0 ||
            strncmp(opcode, "EXIT", 4) == 0);
}

/**
 * Check if instruction is a return
 */
inline bool is_return_instruction(Instr* instr) {
    const char* opcode = instr->getOpcodeShort();
    return (strncmp(opcode, "RET", 3) == 0 ||
            strncmp(opcode, "EXIT", 4) == 0);
}

/**
 * Check if opcode matches filter pattern
 * Supports prefix matching (e.g., "FFMA" matches "FFMA.RN")
 */
inline bool opcode_matches(Instr* instr, const char* filter) {
    if (filter == nullptr) return true;
    const char* opcode = instr->getOpcode();
    return (strncmp(opcode, filter, strlen(filter)) == 0);
}

/* ============================================
 * Section 2: Opcode ID Management
 * ============================================ */

class OpcodeIdManager {
public:
    static OpcodeIdManager& instance() {
        static OpcodeIdManager mgr;
        return mgr;
    }
    
    uint32_t get_id(const std::string& opcode) {
        auto it = opcode_to_id_.find(opcode);
        if (it != opcode_to_id_.end()) {
            return it->second;
        }
        uint32_t id = next_id_++;
        opcode_to_id_[opcode] = id;
        id_to_opcode_[id] = opcode;
        return id;
    }
    
    const char* get_opcode(uint32_t id) {
        auto it = id_to_opcode_.find(id);
        if (it != id_to_opcode_.end()) {
            return it->second.c_str();
        }
        return "UNKNOWN";
    }
    
    void for_each(std::function<void(uint32_t id, const std::string& op)> cb) {
        for (const auto& pair : opcode_to_id_) {
            cb(pair.second, pair.first);
        }
    }

private:
    OpcodeIdManager() : next_id_(0) {}
    std::map<std::string, uint32_t> opcode_to_id_;
    std::map<uint32_t, std::string> id_to_opcode_;
    uint32_t next_id_;
};

/* ============================================
 * Section 3: Function Tracking
 * ============================================ */

inline std::unordered_set<CUfunction>& nvbpf_instrumented_functions() {
    static std::unordered_set<CUfunction> set;
    return set;
}

inline bool nvbpf_is_instrumented(CUfunction func) {
    return nvbpf_instrumented_functions().count(func) > 0;
}

inline void nvbpf_mark_instrumented(CUfunction func) {
    nvbpf_instrumented_functions().insert(func);
}

/* ============================================
 * Section 4: Hook Attachment
 * ============================================ */

/**
 * Attach a single hook entry to an instruction
 */
inline void nvbpf_attach_hook_to_instr(Instr* instr, 
                                       const NvBpfHookEntry* hook,
                                       bool with_addr = false) {
    // Insert call to the device function
    nvbit_insert_call(instr, hook->device_func_name, IPOINT_BEFORE);
    
    // Always pass predicate value first
    nvbit_add_call_arg_guard_pred_val(instr);
    
    // For memory hooks, add address argument
    if (with_addr && (hook->type == HOOK_TRACEPOINT_MEM_LOAD || 
                      hook->type == HOOK_TRACEPOINT_MEM_STORE)) {
        nvbit_add_call_arg_mref_addr64(instr, 0);
    }
    
    // For instruction hooks, add opcode ID
    if (hook->type == HOOK_TRACEPOINT_INSTR_ALL) {
        uint32_t opcode_id = OpcodeIdManager::instance().get_id(instr->getOpcode());
        nvbit_add_call_arg_const_val32(instr, opcode_id);
    }
}

/**
 * Instrument a function based on registered hooks
 */
inline void nvbpf_instrument_function(CUcontext ctx, CUfunction func) {
    if (nvbpf_is_instrumented(func)) {
        return;  // Already instrumented
    }
    
    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, func);
    if (instrs.empty()) return;
    
    NvBpfHookRegistry& registry = nvbpf_get_hook_registry();
    
    // Process each hook type
    registry.for_each([&](NvBpfHookEntry* hook) {
        switch (hook->type) {
            case HOOK_KPROBE_ENTRY:
                // Attach to first instruction only
                nvbpf_attach_hook_to_instr(instrs[0], hook);
                break;
                
            case HOOK_KRETPROBE_EXIT:
                // Attach to all return instructions
                for (auto* instr : instrs) {
                    if (is_return_instruction(instr)) {
                        nvbpf_attach_hook_to_instr(instr, hook);
                    }
                }
                break;
                
            case HOOK_TRACEPOINT_MEM_LOAD:
                // Attach to all load instructions
                for (auto* instr : instrs) {
                    if (instr->isLoad() && 
                        instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {
                        nvbpf_attach_hook_to_instr(instr, hook, true);
                    }
                }
                break;
                
            case HOOK_TRACEPOINT_MEM_STORE:
                // Attach to all store instructions
                for (auto* instr : instrs) {
                    if (instr->isStore()) {
                        nvbpf_attach_hook_to_instr(instr, hook, true);
                    }
                }
                break;
                
            case HOOK_TRACEPOINT_INSTR_ALL:
                // Attach to every instruction
                for (auto* instr : instrs) {
                    nvbpf_attach_hook_to_instr(instr, hook);
                }
                break;
                
            case HOOK_TRACEPOINT_INSTR_BRANCH:
                // Attach to branch instructions
                for (auto* instr : instrs) {
                    if (is_branch_instruction(instr)) {
                        nvbpf_attach_hook_to_instr(instr, hook);
                    }
                }
                break;
                
            case HOOK_TRACEPOINT_INSTR_OPCODE:
                // Attach to specific opcodes
                for (auto* instr : instrs) {
                    if (opcode_matches(instr, hook->filter)) {
                        nvbpf_attach_hook_to_instr(instr, hook);
                    }
                }
                break;
                
            default:
                break;
        }
    });
    
    nvbpf_mark_instrumented(func);
}

/**
 * Instrument function and all related functions (device functions called by kernel)
 */
inline void nvbpf_instrument_function_recursive(CUcontext ctx, CUfunction func) {
    // Get related functions
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);
    
    for (auto f : related) {
        nvbpf_instrument_function(ctx, f);
    }
}

/* ============================================
 * Section 5: Convenience Functions for Tool Authors
 * ============================================ */

/**
 * Main entry point - call this in nvbit_at_cuda_event for kernel launches
 */
inline void nvbpf_attach_hooks(CUcontext ctx, CUfunction func) {
    nvbpf_instrument_function_recursive(ctx, func);
    nvbit_enable_instrumented(ctx, func, true);
}

/**
 * Print registered hooks (for debugging)
 */
inline void nvbpf_print_hooks() {
    printf("[NVBPF] Registered hooks:\n");
    nvbpf_get_hook_registry().for_each([](NvBpfHookEntry* hook) {
        const char* type_str = "unknown";
        switch (hook->type) {
            case HOOK_KPROBE_ENTRY: type_str = "kprobe/entry"; break;
            case HOOK_KRETPROBE_EXIT: type_str = "kretprobe/exit"; break;
            case HOOK_TRACEPOINT_MEM_LOAD: type_str = "tracepoint/mem/load"; break;
            case HOOK_TRACEPOINT_MEM_STORE: type_str = "tracepoint/mem/store"; break;
            case HOOK_TRACEPOINT_INSTR_ALL: type_str = "tracepoint/instr/all"; break;
            case HOOK_TRACEPOINT_INSTR_BRANCH: type_str = "tracepoint/instr/branch"; break;
            case HOOK_TRACEPOINT_INSTR_OPCODE: type_str = "tracepoint/instr/opcode"; break;
            default: break;
        }
        printf("  [%s] %s -> %s", type_str, hook->name, hook->device_func_name);
        if (hook->filter) {
            printf(" (filter: %s)", hook->filter);
        }
        printf("\n");
    });
}

/**
 * Print opcode statistics (for debugging)
 */
inline void nvbpf_print_opcodes() {
    printf("[NVBPF] Observed opcodes:\n");
    OpcodeIdManager::instance().for_each([](uint32_t id, const std::string& op) {
        printf("  [%u] %s\n", id, op.c_str());
    });
}

} // namespace nvbpf

// C-style wrapper for simple usage
inline void nvbpf_attach_hooks(CUcontext ctx, CUfunction func) {
    nvbpf::nvbpf_attach_hooks(ctx, func);
}
