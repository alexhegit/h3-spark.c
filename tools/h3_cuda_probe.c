#include "h3_cuda.h"

#include <stdio.h>

int main(void) {
    h3_device_info info;
    char error[256];
    if (!h3_cuda_probe(&info, error, sizeof(error))) {
        fprintf(stderr, "probe failed: %s\n", error);
        return 1;
    }
    printf("CUDA device: %s (%s)\n", info.name, info.architecture);
    printf("  unified memory: %s\n", info.unified_memory ? "yes" : "no");
    printf("  tensor fast path: %s\n", info.metal4 ? "yes" : "no");
    printf("  sm encoding: %d\n", info.apple_gpu_family);
    return 0;
}
