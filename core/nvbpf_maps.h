/*
 * NV-BPF: eBPF-style Wrapper for NVBit
 * Map Definitions (ARRAY, HASH, PERCPU, RINGBUF)
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <stdint.h>
#include <stdio.h>
#include "nvbpf_types.h"

/* ============================================
 * Section 1: BPF_ARRAY - Fixed-size Array Map
 * ============================================ */

/**
 * Safe array map with bounds checking
 * Similar to BPF_MAP_TYPE_ARRAY
 */
template <typename V, int MAX_ENTRIES>
struct BpfArrayMap {
    V data[MAX_ENTRIES];
    uint64_t dropped_accesses;  // Safety counter
    
    __host__ __device__ BpfArrayMap() : dropped_accesses(0) {
        // Zero-initialize data
        for (int i = 0; i < MAX_ENTRIES; i++) {
            data[i] = V{};
        }
    }
    
    /**
     * Lookup element by index
     * Returns nullptr if out of bounds
     */
    __host__ __device__ V* lookup(uint32_t index) {
        if (index >= MAX_ENTRIES) {
            return nullptr;
        }
        return &data[index];
    }
    
    /**
     * Update element (safe, with bounds check)
     * Returns 0 on success, -1 on error
     */
    __device__ int update(uint32_t index, const V& value) {
        if (index >= MAX_ENTRIES) {
            atomicAdd((unsigned long long*)&dropped_accesses, 1ULL);
            return NVBPF_ERR_INVALID;
        }
        data[index] = value;
        return NVBPF_OK;
    }
    
    /**
     * Atomic increment (for numeric types)
     */
    __device__ void atomic_inc(uint32_t index) {
        if (index < MAX_ENTRIES) {
            static_assert(sizeof(V) == 4 || sizeof(V) == 8,
                "atomic_inc only supports 32/64-bit integers");
            if constexpr (sizeof(V) == 8) {
                atomicAdd((unsigned long long*)&data[index], 1ULL);
            } else {
                atomicAdd((unsigned int*)&data[index], 1U);
            }
        } else {
            atomicAdd((unsigned long long*)&dropped_accesses, 1ULL);
        }
    }
    
    /**
     * Atomic add (for numeric types)
     */
    __device__ void atomic_add(uint32_t index, V value) {
        if (index < MAX_ENTRIES) {
            if constexpr (sizeof(V) == 8) {
                atomicAdd((unsigned long long*)&data[index], (unsigned long long)value);
            } else {
                atomicAdd((unsigned int*)&data[index], (unsigned int)value);
            }
        } else {
            atomicAdd((unsigned long long*)&dropped_accesses, 1ULL);
        }
    }
    
    /**
     * Reset all elements to zero
     */
    __host__ void reset() {
        for (int i = 0; i < MAX_ENTRIES; i++) {
            data[i] = V{};
        }
        dropped_accesses = 0;
    }
    
    /**
     * Get map size
     */
    __host__ __device__ constexpr int size() const { return MAX_ENTRIES; }
};

// Convenience macro
#define BPF_ARRAY(name, value_type, max_entries) \
    __managed__ BpfArrayMap<value_type, max_entries> name

/* ============================================
 * Section 2: BPF_HASH - Hash Map
 * Uses open addressing with linear probing
 * ============================================ */

/**
 * Hash map entry
 */
template <typename K, typename V>
struct BpfHashEntry {
    K key;
    V value;
    uint8_t occupied : 1;
    uint8_t deleted  : 1;
    uint8_t _pad     : 6;
};

/**
 * Simple hash function (FNV-1a)
 */
__device__ __host__ __forceinline__ uint32_t bpf_hash(const void* data, size_t len) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= 16777619u;
    }
    return hash;
}

/**
 * Hash map with open addressing
 * Similar to BPF_MAP_TYPE_HASH
 */
template <typename K, typename V, int MAX_ENTRIES>
struct BpfHashMap {
    BpfHashEntry<K, V> entries[MAX_ENTRIES];
    uint64_t collisions;
    uint64_t dropped;
    
    __host__ __device__ BpfHashMap() : collisions(0), dropped(0) {
        for (int i = 0; i < MAX_ENTRIES; i++) {
            entries[i].occupied = 0;
            entries[i].deleted = 0;
        }
    }
    
    /**
     * Lookup by key
     * Returns nullptr if not found
     */
    __device__ V* lookup(const K* key) {
        if (key == nullptr) return nullptr;
        
        uint32_t hash = bpf_hash(key, sizeof(K));
        uint32_t idx = hash % MAX_ENTRIES;
        
        for (int probe = 0; probe < MAX_ENTRIES; probe++) {
            uint32_t i = (idx + probe) % MAX_ENTRIES;
            
            if (!entries[i].occupied && !entries[i].deleted) {
                // Empty slot, key not found
                return nullptr;
            }
            
            if (entries[i].occupied) {
                // Compare keys byte-by-byte
                bool match = true;
                const uint8_t* k1 = (const uint8_t*)key;
                const uint8_t* k2 = (const uint8_t*)&entries[i].key;
                for (size_t j = 0; j < sizeof(K); j++) {
                    if (k1[j] != k2[j]) { match = false; break; }
                }
                if (match) {
                    return &entries[i].value;
                }
            }
        }
        return nullptr;
    }
    
    /**
     * Insert/update key-value pair
     * Returns 0 on success, negative on error
     */
    __device__ int update(const K* key, const V* value) {
        if (key == nullptr || value == nullptr) return NVBPF_ERR_NULLPTR;
        
        uint32_t hash = bpf_hash(key, sizeof(K));
        uint32_t idx = hash % MAX_ENTRIES;
        int first_deleted = -1;
        
        for (int probe = 0; probe < MAX_ENTRIES; probe++) {
            uint32_t i = (idx + probe) % MAX_ENTRIES;
            
            if (probe > 0) {
                atomicAdd((unsigned long long*)&collisions, 1ULL);
            }
            
            // Track first deleted slot for potential insertion
            if (entries[i].deleted && first_deleted < 0) {
                first_deleted = i;
            }
            
            if (!entries[i].occupied && !entries[i].deleted) {
                // Empty slot - insert here (or at first_deleted if found)
                int target = (first_deleted >= 0) ? first_deleted : i;
                entries[target].key = *key;
                entries[target].value = *value;
                entries[target].occupied = 1;
                entries[target].deleted = 0;
                return NVBPF_OK;
            }
            
            if (entries[i].occupied) {
                // Check if same key - update value
                bool match = true;
                const uint8_t* k1 = (const uint8_t*)key;
                const uint8_t* k2 = (const uint8_t*)&entries[i].key;
                for (size_t j = 0; j < sizeof(K); j++) {
                    if (k1[j] != k2[j]) { match = false; break; }
                }
                if (match) {
                    entries[i].value = *value;
                    return NVBPF_OK;
                }
            }
        }
        
        // Table full
        atomicAdd((unsigned long long*)&dropped, 1ULL);
        return NVBPF_ERR_NO_SPACE;
    }
    
    /**
     * Delete by key
     */
    __device__ int remove(const K* key) {
        if (key == nullptr) return NVBPF_ERR_NULLPTR;
        
        uint32_t hash = bpf_hash(key, sizeof(K));
        uint32_t idx = hash % MAX_ENTRIES;
        
        for (int probe = 0; probe < MAX_ENTRIES; probe++) {
            uint32_t i = (idx + probe) % MAX_ENTRIES;
            
            if (!entries[i].occupied && !entries[i].deleted) {
                return NVBPF_ERR_NOT_FOUND;
            }
            
            if (entries[i].occupied) {
                bool match = true;
                const uint8_t* k1 = (const uint8_t*)key;
                const uint8_t* k2 = (const uint8_t*)&entries[i].key;
                for (size_t j = 0; j < sizeof(K); j++) {
                    if (k1[j] != k2[j]) { match = false; break; }
                }
                if (match) {
                    entries[i].occupied = 0;
                    entries[i].deleted = 1;
                    return NVBPF_OK;
                }
            }
        }
        return NVBPF_ERR_NOT_FOUND;
    }
    
    /**
     * Get a value or insert default
     */
    __device__ V* lookup_or_init(const K* key, const V* init_value) {
        V* existing = lookup(key);
        if (existing) return existing;
        
        if (update(key, init_value) == NVBPF_OK) {
            return lookup(key);
        }
        return nullptr;
    }
};

// Convenience macro
#define BPF_HASH(name, key_type, value_type, max_entries) \
    __managed__ BpfHashMap<key_type, value_type, max_entries> name

/* ============================================
 * Section 3: BPF_PERCPU_ARRAY - Per-SM Array
 * ============================================ */

#define NVBPF_MAX_SMS 256

/**
 * Per-SM array map - each SM gets its own copy
 * Reduces contention for counters
 */
template <typename V, int ENTRIES_PER_SM>
struct BpfPercpuArrayMap {
    V data[NVBPF_MAX_SMS][ENTRIES_PER_SM];
    
    __host__ __device__ BpfPercpuArrayMap() {
        for (int sm = 0; sm < NVBPF_MAX_SMS; sm++) {
            for (int i = 0; i < ENTRIES_PER_SM; i++) {
                data[sm][i] = V{};
            }
        }
    }
    
    /**
     * Lookup for current SM
     */
    __device__ V* lookup(uint32_t index) {
        if (index >= ENTRIES_PER_SM) return nullptr;
        uint32_t sm = bpf_get_current_sm_id();
        if (sm >= NVBPF_MAX_SMS) return nullptr;
        return &data[sm][index];
    }
    
    /**
     * Lookup for specific SM (host access)
     */
    __host__ V* lookup_sm(uint32_t sm_id, uint32_t index) {
        if (sm_id >= NVBPF_MAX_SMS || index >= ENTRIES_PER_SM) return nullptr;
        return &data[sm_id][index];
    }
    
    /**
     * Atomic increment for current SM
     */
    __device__ void atomic_inc(uint32_t index) {
        V* ptr = lookup(index);
        if (ptr) {
            if constexpr (sizeof(V) == 8) {
                atomicAdd((unsigned long long*)ptr, 1ULL);
            } else {
                atomicAdd((unsigned int*)ptr, 1U);
            }
        }
    }
    
    /**
     * Sum across all SMs (host only)
     */
    __host__ V sum(uint32_t index) {
        V total{};
        for (int sm = 0; sm < NVBPF_MAX_SMS; sm++) {
            if (index < ENTRIES_PER_SM) {
                total += data[sm][index];
            }
        }
        return total;
    }
    
    /**
     * Reset all data
     */
    __host__ void reset() {
        for (int sm = 0; sm < NVBPF_MAX_SMS; sm++) {
            for (int i = 0; i < ENTRIES_PER_SM; i++) {
                data[sm][i] = V{};
            }
        }
    }
};

// Forward declaration for helper
__device__ __forceinline__ uint32_t bpf_get_current_sm_id();

// Convenience macro
#define BPF_PERCPU_ARRAY(name, value_type, entries_per_sm) \
    __managed__ BpfPercpuArrayMap<value_type, entries_per_sm> name

/* ============================================
 * Section 4: BPF_RINGBUF - Ring Buffer
 * Lock-free GPU to CPU data channel
 * ============================================ */

/**
 * Ring buffer for streaming data from GPU to CPU
 * Uses a simple producer (GPU) / consumer (CPU) model
 */
template <typename T, int CAPACITY>
struct BpfRingBufMap {
    T buffer[CAPACITY];
    volatile uint64_t head;  // GPU writes here (producer)
    volatile uint64_t tail;  // CPU reads here (consumer)
    uint64_t dropped;
    
    __host__ __device__ BpfRingBufMap() : head(0), tail(0), dropped(0) {}
    
    /**
     * Reserve space and get pointer to write (GPU side)
     * Returns nullptr if buffer is full
     */
    __device__ T* reserve() {
        uint64_t h = atomicAdd((unsigned long long*)&head, 1ULL);
        uint64_t t = tail;  // relaxed read is fine
        
        // Check if we've wrapped around and caught up to tail
        if (h - t >= CAPACITY) {
            // Buffer full - drop this entry
            atomicAdd((unsigned long long*)&dropped, 1ULL);
            return nullptr;
        }
        
        return &buffer[h % CAPACITY];
    }
    
    /**
     * Submit reserved entry (currently a no-op, data visible immediately)
     */
    __device__ void submit(T* entry) {
        // Memory fence to ensure write is visible
        __threadfence_system();
    }
    
    /**
     * Output helper - reserve, write, submit in one call
     */
    __device__ int output(const T* data) {
        T* slot = reserve();
        if (slot == nullptr) return NVBPF_ERR_NO_SPACE;
        *slot = *data;
        submit(slot);
        return NVBPF_OK;
    }
    
    /**
     * Read available entries (CPU side)
     * Callback receives each entry
     */
    template <typename Callback>
    __host__ uint64_t consume(Callback&& callback) {
        uint64_t h = head;
        uint64_t t = tail;
        uint64_t count = 0;
        
        while (t < h) {
            callback(&buffer[t % CAPACITY]);
            t++;
            count++;
        }
        
        tail = t;
        return count;
    }
    
    /**
     * Check how many entries are available
     */
    __host__ uint64_t available() const {
        return head - tail;
    }
    
    /**
     * Reset buffer
     */
    __host__ void reset() {
        head = 0;
        tail = 0;
        dropped = 0;
    }
};

// Convenience macro
#define BPF_RINGBUF(name, entry_type, capacity) \
    __managed__ BpfRingBufMap<entry_type, capacity> name

/* ============================================
 * Section 5: Legacy Compatibility
 * ============================================ */

// Alias for backward compatibility with existing SafeBpfMap usage
#define BPF_MAP_DEF(name, type, size) BPF_ARRAY(name, type, size)
