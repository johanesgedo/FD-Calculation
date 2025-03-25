/*
    MIT License

    Copyright (c) 2023 Johanes_Gedo

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <unordered_map>
#include <iostream>
#include <string.h>
#include <cstring>
#include "helper_cuda.h"

/// Data structure for storing memory information
struct MemInfo {
    size_t size;
    bool isCuda;  /// True if CUDA, false if host (malloc)
};

/// Structure for tracking CUDA event and attribute allocations
struct EventInfo {
    bool isCuda;
};
struct AttributeInfo {
    int value;
};

/// Map for tracking all memory allocations
std::unordered_map<void*, MemInfo> memMap;

/// Map for storing CUDA events and device attributes
std::unordered_map<cudaEvent_t, EventInfo> eventMap;
std::unordered_map<int, AttributeInfo> attributeMap;

/// Helper function to record allocations
void trackMemory(void* ptr, size_t size, bool isCuda) {
    if (ptr) {
        /// Only track if not already on the map
        if (memMap.find(ptr) == memMap.end()) {
            memMap[ptr] = {size, isCuda};
            /// std::cout << "[TRACK] Memory usage updated: " << size << " bytes at " << ptr << " (" << (isCuda ? "CUDA" : "Host") << ")" << std::endl;
        }
    }
}

/// Auxiliary function to record exemptions
void untrackMemory(void* ptr) {
    if (memMap.count(ptr)) {
        /// std::cout << "[FREE] Freed memory at " << ptr << " ("<< (memMap[ptr].isCuda ? "CUDA" : "Host") << ")" << std::endl;
        memMap.erase(ptr);
    }
    else {
        std::cerr << "[WARNING] Attempted to free untracked memory at " << ptr << std::endl;
    }
}

/// Function to record CUDA events
void trackEvent(cudaEvent_t event) {
    eventMap[event] = {true};
}

/// Function to delete events from tracking
void untrackEvent(cudaEvent_t event) {
    if (eventMap.count(event)) {
        eventMap.erase(event);
    }
}

/// cudaGetDeviceCount wrapper with error checking
void SafecudaGetDeviceCount(int* count) {
    checkCudaErrors(cudaGetDeviceCount(count));
    std::cout << "[INFO] Number of CUDA devices: " << *count << std::endl;
}

/// cudaMalloc wrapper
void* SafecudaMalloc(void** devPtr, size_t size) {
    checkCudaErrors(cudaMalloc(devPtr, size));
    if (*devPtr) {
        trackMemory(*devPtr, size, true);
        std::cout << "[DEBUG] Allocated " << size << " bytes at " << *devPtr << " (CUDA) in function safeCudaMalloc" << std::endl;
    }
    return *devPtr;
}

/// cudaMallocAsync wrapper
void SafecudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    checkCudaErrors(cudaMallocAsync(devPtr, size, stream));
    if (*devPtr) {
        trackMemory(*devPtr, size, true);
        std::cout << "[DEBUG] Allocated " << size << " bytes at " << *devPtr << " (CUDA) in function safeCudaMallocAsync" << std::endl;
    }
    /// return *devPtr;
}

/// cudaMemcpy wrapper
void SafecudaMemcpy(void* devPtr, const void* src, size_t size, cudaMemcpyKind kind) {
    checkCudaErrors(cudaMemcpy(devPtr, src, size, kind));

    /// Memory track only if not previously recorded
    if (memMap.find(devPtr) == memMap.end()) {
        bool isCuda = (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice);
        trackMemory(devPtr, size, isCuda);
    }
}


/// cudaMemcpyAsync wrapper
void SafecudaMemcpyAsync(void* devPtr, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    checkCudaErrors(cudaMemcpyAsync(devPtr, src, size, kind, stream));

    /// Memory track only if not previously recorded
    if (memMap.find(devPtr) == memMap.end()) {
        bool isCuda = (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice);
        trackMemory(devPtr, size, isCuda);
    }
}

/// cudaMemset wrapper
void* SafecudaMemset(void** devPtr, int value, size_t size) {
    checkCudaErrors(cudaMemset(*devPtr, value, size));
    /// Check if this pointer is a CUDA allocation, then track if it is not already there.
    if (memMap.find(*devPtr) != memMap.end()) {
        trackMemory(*devPtr, size, true);
    }
    return *devPtr;
}

/// cudaMemsetAsync wrapper
void* SafecudaMemsetAsync(void** devPtr, int value, size_t size, cudaStream_t stream) {
    checkCudaErrors(cudaMemsetAsync(*devPtr, value, size, stream));
    /// Check if this pointer is a CUDA allocation, then track if it is not already there.
    if (memMap.find(*devPtr) != memMap.end()) {
        trackMemory(*devPtr, size, true);
    }
    return *devPtr;
}

/// cudaHostAlloc wrapper
void SafecudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    checkCudaErrors(cudaHostAlloc(ptr, size, flags));
    trackMemory(*ptr, size, false); /// False because this is host memory
    std::cout << "[DEBUG] Allocated " << size << " bytes at " << *ptr << " (Pinned Host) using safeCudaHostAlloc." << std::endl;
}

/// cudaFreeHost wrapper
void SafecudaFreeHost(void* ptr) {
    if (ptr) {
        checkCudaErrors(cudaFreeHost(ptr));
        untrackMemory(ptr);
    }
}

// cudaFree wrapper
void SafecudaFree(void* devPtr) {
    if (devPtr) {
        checkCudaErrors(cudaFree(devPtr));
        untrackMemory(devPtr);
    }
}

/// cudaFreeAsync wrapper
void SafecudaFreeAsync(void* devPtr, cudaStream_t stream) {
    if (devPtr) {
        checkCudaErrors(cudaFreeAsync(devPtr, stream));
        untrackMemory(devPtr);
    }
}

/// malloc wrapper
void* Safemalloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        std::cerr << "Error: Host memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
    trackMemory(ptr, size, false);
    return ptr;
}

// Safe wrapper untuk memcpy (CPU memory copy)
void Safememcpy(void* dst, const void* src, size_t size) {
    if (!dst || !src) {
        std::cerr << "[ERROR] safeMemcpy failed: NULL pointer detected." << std::endl;
        return;
    }
    std::memcpy(dst, src, size);

    // Memory track only if not previously recorded
    if (memMap.find(dst) == memMap.end()) {
        trackMemory(dst, size, false);
    }

    std::cout << "[DEBUG] Copied " << size << " bytes from " << src << " to " << dst << " using safeMemcpy." << std::endl;
}

/// free wrapper
void Safefree(void* ptr) {
    if (ptr) {
        free(ptr);
        untrackMemory(ptr);
    }
}

/// cudaStreamCreate wrapper
cudaStream_t SafecudaStreamCreate() {
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    return stream;
}

/// cudaStreamCreate wrapper with parameters
void SafecudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    checkCudaErrors(cudaStreamCreateWithFlags(stream, flags));
    trackMemory((void*)(*stream), sizeof(cudaStream_t), true);
}

/// cudaStreamDestroy Wrapper
void SafecudaStreamDestroy(cudaStream_t stream) {
    if (stream) {
        checkCudaErrors(cudaStreamDestroy(stream));
        untrackMemory((void*)stream);
    }
}

/// cudaStreamSynchronize wrapper
void SafecudaStreamSynchronize(cudaStream_t stream) {
    if (!stream) {
        std::cerr << "Error: Attempt to synchronize a NULL stream!" << std::endl;
        return;
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
}

/// cudaEventCreate wrapper
cudaEvent_t SafecudaEventCreate() {
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));
    trackEvent(event);
    trackMemory((void*)event, sizeof(cudaEvent_t), true);
    return event;
}

/// cudaEventRecord wrapper
void SafecudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
    checkCudaErrors(cudaEventRecord(event, stream));
}

/// cudaEventSynchronize wrapper
void SafecudaEventSynchronize(cudaEvent_t event) {
    checkCudaErrors(cudaEventSynchronize(event));
}

// cudaEventElapsedTime wrapper
void SafecudaEventElapsedTime(float* time, cudaEvent_t start, cudaEvent_t stop) {
    checkCudaErrors(cudaEventElapsedTime(time, start, stop));
}

/// cudaEventDestroy wrapper
void SafecudaEventDestroy(cudaEvent_t event) {
    if (event) {
        checkCudaErrors(cudaEventDestroy(event));
        untrackMemory(event);
    }
}

/// cudaDeviceSynchronize wrapper
void SafecudaDeviceSynchronize() {
    checkCudaErrors(cudaDeviceSynchronize());
}

// cudaDeviceGetAttribute wrapper with tracking
void SafecudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    checkCudaErrors(cudaDeviceGetAttribute(value, attr, device));
    attributeMap[device] = {*value};
}

/// Helper function to print memory reports
void printMemoryLeak() {
    if (memMap.empty()) {
        std::cout << "No memory leaks detected." << std::endl;
        return;
    }

    std::cout << "Memory Leak Detection:" << std::endl;
    for (auto& pair : memMap) {
        void* ptr = pair.first;
        MemInfo info = pair.second;
        if (info.isCuda) {
            std::cout << "  - CUDA Memory Leak: " << ptr << " | Size: " << info.size << " bytes" << std::endl;
            cudaFree(ptr);  /// Free up CUDA memory
        }
        else {
            std::cout << "  - Host Memory Leak: " << ptr << " | Size: " << info.size << " bytes" << std::endl;
            free(ptr);  /// Free up CPU memory
        }
    }
    memMap.clear();
    std::cout << "All memory leaks have been freed!" << std::endl;
}



