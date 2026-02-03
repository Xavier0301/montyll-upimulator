#ifndef DPU_TINYMAT_H
#define DPU_TINYMAT_H

/* 
    Lightweight matrix class for DPUs 
    Columns can only be a power of 2,
        which allows for fast index calculation
        given the 8-bit mul constraint of DPUs
        => lmat stands for log mat
*/

#include "tinymem.h"

#define LMAT_TYPE_(symbol) lmat_##symbol##_
#define LMAT_TYPE(symbol) lmat_##symbol

// cols = stride
#define DEFINE_LMAT_STRUCT(symbol) \
    typedef struct LMAT_TYPE_(symbol) { \
        symbol* data; \
        u8 rows; \
        u8 log_cols; \
    } LMAT_TYPE(symbol)

DEFINE_LMAT_STRUCT(u32);
DEFINE_LMAT_STRUCT(u16);
DEFINE_LMAT_STRUCT(u8);

#define LMAT(t, i, j) ((t).data[((i) << (t).log_cols) + (j)])
#define LMATP(t, i, j) ((t).data + ((i) << (t).log_cols) + (j))

#define LMAT_COUNT(m) ((m)->rows << (m)->log_cols)

#define LMAT_PRINT(m, rows, cols) \
    do { \
        for(size_t i = 0; i < rows; ++i) { \
            for(size_t j = 0; j < cols; ++j) \
                printf("%u ", LMAT(m, i, j)); \
            printf("\n"); \
        } \
    } while(0)

#define DEFINE_LMAT_INIT(symbol) \
    static inline \
    void lmat_##symbol##_init(LMAT_TYPE(symbol)* m, u8 rows, u8 log_cols) { \
        (m)->rows = rows; \
        (m)->log_cols = log_cols; \
        (m)->data = (symbol*) wram_alloc_##symbol(rows << log_cols); \
    }

DEFINE_LMAT_INIT(u32);
DEFINE_LMAT_INIT(u16);
DEFINE_LMAT_INIT(u8);

#endif // DPU_TINYMAT_H
