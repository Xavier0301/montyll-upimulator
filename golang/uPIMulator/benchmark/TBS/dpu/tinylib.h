#ifndef _DPU_TINYLIB_H_
#define _DPU_TINYLIB_H_

#include "../tbtc-htm/types.h"

#include "../tbtc-htm/lm_parameters.h"

#include "tinymat.h"

/* Bare bones learning module utility functions */

// ============= PACKED BITARRAY ==============

typedef struct wram_blocks_t_ {
    u32 f_feature_context;
    u32 f_location_context;

    u32 l_location_context;
    u32 l_feature_context;

    u32 o_internal_context;
    u32 o_external_context;
    u32 o_feedforward;

    u32 f_feature_spike_cache;
    u32 f_location_spike_cache;

    u32 l_location_spike_cache;
    u32 l_feature_spike_cache;

    u32 o_internal_spike_cache;
    u32 o_external_spike_cache;
} wram_blocks_t;

#define PRINT_WRAM_BLOCKS(c) \
    printf("WRAM Content:\n"); \
    printf("\t f feature context block:       %u B\n", c.f_feature_context); \
    printf("\t f location context block:      %u B\n", c.f_location_context); \
    printf("\t l location context block:      %u B\n", c.l_location_context); \
    printf("\t l feature context block:       %u B\n", c.l_feature_context); \
    printf("\t o internal context block:      %u B\n", c.o_internal_context); \
    printf("\t o external context block:      %u B\n", c.o_external_context); \
    printf("\t o feedforward block:           %u B\n", c.o_feedforward); \
    printf("\t f feature spike count block:   %u B\n", c.f_feature_spike_cache); \
    printf("\t f location spike count block:  %u B\n", c.f_location_spike_cache); \
    printf("\t l location spike count block:  %u B\n", c.l_location_spike_cache); \
    printf("\t l feature spike count block:   %u B\n", c.l_feature_spike_cache); \
    printf("\t o internal spike count block:  %u B\n", c.o_internal_spike_cache); \
    printf("\t o external spike count block:  %u B\n", c.o_external_spike_cache); 



// ============= PACKED BITARRAY ==============

/** For a packed array of type u32*
 * we gotta leave 32 = 2^5 bits of addressing for the bitarray 
 * 
 * Meaning if u32 index is our address, we do:
 *      word_index = index >> 5
 *      bit_index = index & 0b11111
 *      
 *      bit = GET_BIT(packed[word_index], bit_index)
 * 
 * To set a bit, it is:
 *      elements[word_index] |= (1 << bit_index)
 * 
 * To reset a bit, it is:
 *      elements[word_index] &= ~ (1 << bit_index)
 */

#define BIT_MASK(i) (1U << i)
#define GET_BIT(x, i) ((x >> i) & 1) // 2 instructions

#define GET_BIT_FROM_PACKED32(packed, index) (GET_BIT(packed[index >> 5], index & 0b11111)) // 5 instructions
#define SET_BIT_IN_PACKED32(packed, index) (packed[index >> 5] |= (1U << (index & 0b11111))) // 4-5 instructions
#define RESET_BIT_IN_PACKED32(packed, index) (packed[index >> 5] &= ~ (1U << (index & 0b11111))) // 5-6 instructions






// ============= SEGMENTS ==============

// ----- OUTPUT LAYER -----

// u24
typedef struct __attribute__((packed)) internal_output_index_ {
    u16 cell;
    u8 filler; // unused
} internal_output_index;

// u24
typedef struct __attribute__((packed)) external_output_index_ {
    u16 cell;
    u8 lm_id;
} external_output_index;

// ----- FEATURE LAYER -----

// u24
typedef struct __attribute__((packed)) feature_index_ {
    u16 col;
    u8 cell;
} feature_index;

// ----- LOCATION LAYER -----

// u24
typedef struct __attribute__((packed)) location_index_ {
    u16 col;
    u8 cell;
} location_index;

// ----- ABSTRACT SEGMENT -----

union cell_index {
    internal_output_index internal_output;
    external_output_index external_output;
    feature_index feature;
    location_index location;
};

enum segment_index_type {
    INTERNAL_OUTPUT_INDEX_TYPE,
    EXTERNAL_OUTPUT_INDEX_TYPE,
    FEATURE_INDEX_TYPE,
    LOCATION_INDEX_TYPE
};

// u32
typedef struct __attribute__((packed)) segment_data_ {
    union cell_index index;
    u8 permanence;
} segment_data;

#define CONNECTIONS_PER_SEGMENT 40

typedef struct __attribute__((aligned(8))) segment_t_ {
    segment_data connections[CONNECTIONS_PER_SEGMENT]; // each connections takes 4 bytes, we "cast" it to the proper x_segment_data in the code
    u8 num_connections;
    // u8 connection_count; // used for learning, to know which segment was spiking
} segment_t;





// ============= HTM ==============

void htm_prediction_integrate_context_LOCATION(
    u32* output_num_spiking_segments, 
    segment_t* context_pointer, u8* spike_count_pointer, 
    u32* incident_activity, lmat_u32* optional_external_incident_activity,
    u8 num_segments, u8 permanence_threshold, u8 segment_spiking_threshold
);

void htm_prediction_integrate_context_FEATURE(
    u32* output_num_spiking_segments, 
    segment_t* context_pointer, u8* spike_count_pointer, 
    u32* incident_activity, lmat_u32* optional_external_incident_activity,
    u8 num_segments, u8 permanence_threshold, u8 segment_spiking_threshold
);

void htm_prediction_integrate_context_INTERNAL_OUTPUT(
    u32* output_num_spiking_segments, 
    segment_t* context_pointer, u8* spike_count_pointer, 
    u32* incident_activity, lmat_u32* optional_external_incident_activity,
    u8 num_segments, u8 permanence_threshold, u8 segment_spiking_threshold
);

void htm_prediction_integrate_context_EXTERNAL_OUTPUT(
    u32* output_num_spiking_segments, 
    segment_t* context_pointer, u8* spike_count_pointer, 
    u32* incident_activity, lmat_u32* optional_external_incident_activity,
    u8 num_segments, u8 permanence_threshold, u8 segment_spiking_threshold
);

__attribute__((always_inline))
void htm_learning_pick_winner_cell(
    u32* winning_cell, u32* winning_spike_count, 
    u8* spike_count_pointer, 
    u32 segments, u32 cells
);

__attribute__((always_inline))
i32 htm_learning_adjust_permanences(
    u32 cell, 
    segment_t* context_pointer, u8* spike_count_pointer,
    u32* incident_activity, lmat_u32* optional_external_incident_activity,
    u32 cell_is_active, u32 cell_is_predicted,
    u32 col_active_and_unpredicted, u32 winning_cell,
    enum segment_index_type index_type, u8 num_segments,
    htm_params_t* htm_p, u32 enable_decay
);

__attribute__((always_inline))
u32 htm_connections_check(
    segment_t* context_pointer, 
    enum segment_index_type index_type, u8 num_segments,
    u32 connections_min, u32 connections_max,
    u32 col_max, u32 cell_max,
    u32 lm_max,
    u32 debug_print,
    u32 col, u32 cell, u32 tasklet
);

__attribute__((always_inline))
u16 find_kth_largest_from_score_counts(u16* score_counts, u32 max_score, u32 k);

#endif /* _TINYLIB_H_ */
