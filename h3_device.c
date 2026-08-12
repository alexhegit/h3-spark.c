#include "h3_device.h"

#ifdef __APPLE__
#include "h3_metal.h"
#else
#include "h3_cuda.h"
#endif

int h3_device_probe(h3_device_info *info, char *error, size_t error_size) {
#ifdef __APPLE__
    return h3_metal_probe(info, error, error_size);
#else
    return h3_cuda_probe(info, error, error_size);
#endif
}
