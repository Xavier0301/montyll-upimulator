#ifndef _COMMON_H_
#define _COMMON_H_

#include "../tbtc-htm/types.h"
#include "../tbtc-htm/lm_parameters.h"

#define ROUND_UP_TO_MULTIPLE_OF_8(x)    ((((x) + 7) >> 3) << 3)
#define ROUND_DOWN_TO_MULTIPLE_OF_8(x)  (((x) >> 3) << 3)

typedef struct dpu_model_params_t_ {
    output_layer_params_t output;
    feature_layer_params_t features;
    location_layer_params_t location;
} dpu_model_params_t;

#define PRINT_MODEL_PARAMS(p) \
    printf("== Learning module params:\n"); \
    feature_layer_print_params(p.features); \
    location_layer_print_params(p.location); \
    output_layer_print_params(p.output); 

typedef struct mram_content_t_ {
    u32 f_feature_context;
    u32 f_location_context;
    
    u32 l_location_context;
    u32 l_feature_context;

    u32 o_internal_context;
    u32 o_external_context;
    u32 o_feedforward;

    u32 input_movement;
    u32 input_features; // "active_columns"

    u32 external_o_activity;

    u32 output; 

    u32 f_feature_spike_cache;
    u32 f_location_spike_cache;

    u32 l_location_spike_cache;
    u32 l_feature_spike_cache;

    u32 o_internal_spike_cache;
    u32 o_external_spike_cache;
} mram_content_t;

#define PRINT_MRAM_CONTENT(b, addr) \
    printf("MRAM Content:\n"); \
    printf("\t f feature context:      %u B at %u\n", b.f_feature_context, addr.f_feature_context); \
    printf("\t f location context:     %u B at %u\n", b.f_location_context, addr.f_location_context); \
    printf("\t l location context:     %u B at %u\n", b.l_location_context, addr.l_location_context); \
    printf("\t l feature context:      %u B at %u\n", b.l_feature_context, addr.l_feature_context); \
    printf("\t o internal context:     %u B at %u\n", b.o_internal_context, addr.o_internal_context); \
    printf("\t o external context:     %u B at %u\n", b.o_external_context, addr.o_external_context); \
    printf("\t o feedforward:          %u B at %u\n", b.o_feedforward, addr.o_feedforward); \
    printf("\t input movement:         %u B at %u\n", b.input_movement, addr.input_movement); \
    printf("\t input features:         %u B at %u\n", b.input_features, addr.input_features); \
    printf("\t external out activity:  %u B at %u\n", b.external_o_activity, addr.external_o_activity); \
    printf("\t output:                 %u B at %u\n", b.output, addr.output); \
    printf("\t f feature spike cache:  %u B at %u\n", b.f_feature_spike_cache, addr.f_feature_spike_cache); \
    printf("\t f location spike cache: %u B at %u\n", b.f_location_spike_cache, addr.f_location_spike_cache); \
    printf("\t l location spike cache: %u B at %u\n", b.l_location_spike_cache, addr.l_location_spike_cache); \
    printf("\t l feature spike cache:  %u B at %u\n", b.l_feature_spike_cache, addr.l_feature_spike_cache); \
    printf("\t o internal spike cache: %u B at %u\n", b.o_internal_spike_cache, addr.o_internal_spike_cache); \
    printf("\t o external spike cache: %u B at %u\n", b.o_external_spike_cache, addr.o_external_spike_cache); 
 
// // for measuring cycles/instructions: 
// typedef struct {
//     u64 count;
// } dpu_results_t;

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((b) < (a) ? (a) : (b))

#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)
#define aligned_count(count, size_bytes) (((count * size_bytes) % 8) != 0 ? roundup(count, 8) : count)
#define aligned_size_bytes(count, size_bytes) (aligned_count(count, size_bytes) * size_bytes)

#define NUM_ITERATIONS(num_blocks, num_iterations, block_it) (num_iterations / num_blocks + (num_iterations % num_blocks > block_it ? 1 : 0))

#define TO_KB(bytes) ((bytes) / 1024)
#define TO_MB(bytes) ((bytes) / (1024 * 1024))

#endif
