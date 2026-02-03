#ifndef _DPU_UTILS_H_
#define _DPU_UTILS_H_

#include <stdio.h>

#include "../tbtc-htm/types.h"
#include "tinylib.h"

static inline
u32 count_set_bits(u32 n) {
    u32 count = 0;
    while (n > 0) {
        n &= (n - 1); // This trick removes the lowest set bit
        count++;
    }
    return count;
}

void print_packed_spvec_u32(u32* v, u32 length) {
    u32 count = 0;
    for(u32 i = 0; i < length; ++i) {
        count += count_set_bits(v[i]);
    }

    printf("[%u, %u, %f] ", length << 5, count, ((f32) count) / (f32) (length << 5));
}

void print_spvec_u8(u8* v, u32 length) {
    u32 count = 0;
    for(u32 i = 0; i < length; ++i) {
        count += v[i] == 1;
    }

    printf("[%u, %u, %f] ", length, count, ((f32) count) / (f32) length);
}

#endif /* _DPU_UTILS_H_ */
