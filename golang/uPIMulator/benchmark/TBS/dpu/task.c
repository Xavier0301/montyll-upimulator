/*
* AXPY with multiple tasklets
*
*/
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include "assert.h"
#include "string.h"

#include "../support/common.h"

#if PRINT == 2
#include "../tbtc-htm/lm_parameters.h"
#endif

#include "tinylib.h"
#include "tinymem.h"

#include "utils.h"

#define O_SPIKE_CACHE_BLOCK_NUM_CELLS 8

#define MRAM_READ_COMPONENT(dst_buffer, component) \
    mram_read( \
        (__mram_ptr void*) running_addresses->component, \
        dst_buffer, \
        wram_block_bytes.component \
    )

#define MRAM_WRITE_COMPONENT(src_buffer, component) \
    mram_write( \
        src_buffer, \
        (__mram_ptr void*) running_addresses->component, \
        wram_block_bytes.component \
    )

// Input parameters and output performance counts
__host dpu_model_params_t p;

__host mram_content_t size_bytes;
__host mram_content_t addresses;

wram_blocks_t wram_block_bytes;

mram_content_t outer_strides; 

mram_content_t base_addresses_global[NR_TASKLETS];
mram_content_t running_addresses_global[NR_TASKLETS];

// #if defined(CYCLES) || defined(INSTRUCTIONS)
// __host dpu_results_t DPU_RESULTS[NR_TASKLETS];
// #endif
// Shared wram data
u32* tmp_active_prev; // (cols) for learning step, need to copy overwritten .active of current layer to apply learning rules

// feature layer: 12 KB
u32* f_active; // (cols) packed
u32* f_predicted; // (cols) packed

// location layer: 12 KB
u32* l_active; // (cols) packed
u32* l_predicted; // (cols) packed

// output layer: 1 KB
u32* o_active; // (cells) packed, 128B
u8* o_prediction_scores; // (cells), 1KB
u16* o_prediction_score_counts; // (segments), 24B

lmat_u32 external_o_activity; // (num_lms, cells) packed on axis 2, 128B * num_lms

// input: 1032 bytes
u8* input_buffer; // movement_x (4B), movement_y (4B), input (1024B)
u32 movement_x, movement_y; // copy of input_buffer[0..3] and input_buffer[4..7]
u8* features; // points to input_buffer[8]

// Barrier
BARRIER_INIT(reset_barrier, NR_TASKLETS);
BARRIER_INIT(base_initialization_barrier, NR_TASKLETS);
BARRIER_INIT(full_initialization_barrier, NR_TASKLETS);

BARRIER_INIT(location_act_reset_end_barrier, NR_TASKLETS);
BARRIER_INIT(location_col_act_end_barrier, NR_TASKLETS);
BARRIER_INIT(location_cell_act_end_barrier, NR_TASKLETS);

BARRIER_INIT(feature_predict_end_barrier, NR_TASKLETS);
BARRIER_INIT(feature_cell_act_end_barrier, NR_TASKLETS);

BARRIER_INIT(output_predict_end_barrier, NR_TASKLETS);
BARRIER_INIT(output_cell_act_end_barrier, NR_TASKLETS);
BARRIER_INIT(output_collect_counts_end_barrier, NR_TASKLETS);
BARRIER_INIT(output_find_kth_end_barrier, NR_TASKLETS);
BARRIER_INIT(output_learn_end_barrier, NR_TASKLETS);

extern int fused_kernel(void);
int main(void) { 
    return fused_kernel();
}

int fused_kernel() {
    unsigned int tasklet_id = me();
    if (tasklet_id == 0) { 
        mem_reset(); // Reset the heap
    }

    // Barrier
    barrier_wait(&reset_barrier);
// #if defined(CYCLES) || defined(INSTRUCTIONS)
//     dpu_results_t *result = &DPU_RESULTS[tasklet_id];
//     result->count = 0;
// #endif
// #ifdef CYCLES
//     perfcounter_config(COUNT_CYCLES, true); // Initialize once the cycle counter
// #elif INSTRUCTIONS
//     perfcounter_config(COUNT_INSTRUCTIONS, true); // Initialize once the instruction counter
// #endif


    // This will fail the build if the size isn't a multiple of 8
    _Static_assert((sizeof(segment_t) % 8) == 0, "segment_t size must be a multiple of 8 for DMA");

    if(tasklet_id == 0) {
        addresses.f_feature_context += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.f_location_context += (u32) DPU_MRAM_HEAP_POINTER;

        addresses.l_location_context += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.l_feature_context += (u32) DPU_MRAM_HEAP_POINTER;

        addresses.o_internal_context += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.o_external_context += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.o_feedforward += (u32) DPU_MRAM_HEAP_POINTER;

        addresses.input_movement += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.input_features += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.external_o_activity += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.output += (u32) DPU_MRAM_HEAP_POINTER;

        addresses.f_feature_spike_cache += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.f_location_spike_cache += (u32) DPU_MRAM_HEAP_POINTER;

        addresses.l_location_spike_cache += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.l_feature_spike_cache += (u32) DPU_MRAM_HEAP_POINTER;

        addresses.o_internal_spike_cache += (u32) DPU_MRAM_HEAP_POINTER;
        addresses.o_external_spike_cache += (u32) DPU_MRAM_HEAP_POINTER;

        // per cell blocks of segment data, all need to lead to 8 bytes-aligned addresses and be 8 bytes-aligned sizes
        wram_block_bytes = (wram_blocks_t) {
            .f_feature_context = p.features.feature_segments * sizeof(segment_t),
            .f_location_context = p.features.location_segments * sizeof(segment_t),

            .l_location_context = p.location.location_segments * sizeof(segment_t),
            .l_feature_context = p.location.feature_segments * sizeof(segment_t),

            .o_internal_context = p.output.internal_context_segments * sizeof(segment_t),
            .o_external_context = p.output.external_context_segments * sizeof(segment_t),
            .o_feedforward = sizeof(segment_t),

            .f_feature_spike_cache = p.features.cells * p.features.feature_segments * sizeof(u8),
            .f_location_spike_cache = p.features.cells * p.features.location_segments * sizeof(u8),

            .l_location_spike_cache = p.location.cells * p.location.location_segments * sizeof(u8),
            .l_feature_spike_cache = p.location.cells * p.location.feature_segments * sizeof(u8),

            .o_internal_spike_cache = O_SPIKE_CACHE_BLOCK_NUM_CELLS * p.output.internal_context_segments * sizeof(u8),
            .o_external_spike_cache = O_SPIKE_CACHE_BLOCK_NUM_CELLS * p.output.external_context_segments * sizeof(u8)
        };

        // Alloc shared WRAM buffers
        tmp_active_prev = wram_alloc_u32(p.features.cols);
        
        f_active = wram_alloc_u32(p.features.cols);
        f_predicted = wram_alloc_u32(p.features.cols);

        l_active = wram_alloc_u32(p.location.cols);
        l_predicted = wram_alloc_u32(p.location.cols);

        o_active = wram_alloc_u32(p.output.cells >> 5); // packed in u32 = 2^5 bits
        o_prediction_scores = wram_alloc_u8(p.output.cells);
        o_prediction_score_counts = wram_alloc_u16(p.output.internal_context_segments + p.output.external_context_segments);

        lmat_u32_init(&external_o_activity, p.output.external_lms, p.output.log_external_cells - 5); // packed in u32 = 2^5 bits
        mram_read(
            (__mram_ptr void*) addresses.external_o_activity, 
            external_o_activity.data,
            size_bytes.external_o_activity
        );

        input_buffer = wram_alloc_u8(sizeof(u32) + sizeof(u32) + p.features.cols);

        mram_read_u8(addresses.input_movement, input_buffer, size_bytes.input_movement + size_bytes.input_features);

        memcpy(&movement_x, &input_buffer[0], sizeof(u32));
        memcpy(&movement_y, &input_buffer[4], sizeof(u32));

        features = &input_buffer[8]; // first 8 bytes are movement

#if PRINT == 2
        printf("MRAM content address (size)\n");
        printf("\tf_feature_context      : %u (%u B)\n", addresses.f_feature_context, size_bytes.f_feature_context);
        printf("\tf_location_context     : %u (%u B)\n", addresses.f_location_context, size_bytes.f_location_context);
        printf("\tl_location_context     : %u (%u B)\n", addresses.l_location_context, size_bytes.l_location_context);
        printf("\tl_feature_context      : %u (%u B)\n", addresses.l_feature_context, size_bytes.l_feature_context);
        printf("\to_internal_context     : %u (%u B)\n", addresses.o_internal_context, size_bytes.o_internal_context);
        printf("\to_external_context     : %u (%u B)\n", addresses.o_external_context, size_bytes.o_external_context);
        printf("\to_feedforward          : %u (%u B)\n", addresses.o_feedforward, size_bytes.o_feedforward);
        printf("\tinput_movement         : %u (%u B)\n", addresses.input_movement, size_bytes.input_movement);
        printf("\tinput_features         : %u (%u B)\n", addresses.input_features, size_bytes.input_features);
        printf("\texternal_o_activity    : %u (%u B)\n", addresses.external_o_activity, size_bytes.external_o_activity);
        printf("\toutput                 : %u (%u B)\n", addresses.output, size_bytes.output);
        printf("\tf_feature_spike_cache  : %u (%u B)\n", addresses.f_feature_spike_cache, size_bytes.f_feature_spike_cache);
        printf("\tf_location_spike_cache : %u (%u B)\n", addresses.f_location_spike_cache, size_bytes.f_location_spike_cache);
        printf("\tl_location_spike_cache : %u (%u B)\n", addresses.l_location_spike_cache, size_bytes.l_location_spike_cache);
        printf("\tl_feature_spike_cache  : %u (%u B)\n", addresses.l_feature_spike_cache, size_bytes.l_feature_spike_cache);
        printf("\to_internal_spike_cache : %u (%u B)\n", addresses.o_internal_spike_cache, size_bytes.o_internal_spike_cache);
        printf("\to_external_spike_cache : %u (%u B)\n", addresses.o_external_spike_cache, size_bytes.o_external_spike_cache);

        // feature_layer_print_params(p.features);
        // location_layer_print_params(p.location);
        // output_layer_print_params(p.output);

        printf("DPU WRAM Buffer Sizes:\n");
        printf("\tShared:\n");

        printf("\t\ttmp_active_prev: %u B\n", p.features.cols * sizeof(u32));

        printf("\t\tf_active: %u B\n", p.features.cols * sizeof(u32));
        printf("\t\tf_predicted: %u B\n", p.features.cols * sizeof(u32));

        printf("\t\tl_active: %u B\n", p.location.cols * sizeof(u32));
        printf("\t\tl_predicted: %u B\n", p.location.cols * sizeof(u32));

        printf("\t\to_active: %u B\n", (p.output.cells >> 5) * sizeof(u32)); // packed in u32 = 2^5 bits
        printf("\t\to_prediction_scores: %u B\n", p.output.cells * sizeof(u8));
        printf("\t\to_prediction_score_counts: %u B\n", 
            (p.output.internal_context_segments + p.output.external_context_segments) * sizeof(u16));

        printf("\t\texternal_o_activity: %u B\n", LMAT_COUNT(&external_o_activity) * sizeof(u32)); // packed in u32 = 2^5 bits

        printf("\t\tinput: %u B\n", sizeof(u32) + sizeof(u32) + p.features.cols * sizeof(u8));

        printf("\tPrivate:\n");
        printf("\t\tbuffer1: %u B x %u\n", wram_block_bytes.f_feature_context, NR_TASKLETS);
        printf("\t\tbuffer2: %u B x %u\n", wram_block_bytes.f_location_context, NR_TASKLETS);

        printf("\t\tbuffer3: %u B x %u\n", wram_block_bytes.f_feature_spike_cache, NR_TASKLETS);
        printf("\t\tbuffer4: %u B x %u\n", wram_block_bytes.f_location_spike_cache, NR_TASKLETS);

        printf("DPU WRAM Block Bytes:\n");
        printf("\tf_feature_context      : %u\n", wram_block_bytes.f_feature_context);
        printf("\tf_location_context     : %u\n", wram_block_bytes.f_location_context);
        printf("\tl_location_context     : %u\n", wram_block_bytes.l_location_context);
        printf("\tl_feature_context      : %u\n", wram_block_bytes.l_feature_context);
        printf("\to_internal_context     : %u\n", wram_block_bytes.o_internal_context);
        printf("\to_external_context     : %u\n", wram_block_bytes.o_external_context);
        printf("\to_feedforward          : %u\n", wram_block_bytes.o_feedforward);
        printf("\tf_feature_spike_cache  : %u\n", wram_block_bytes.f_feature_spike_cache);
        printf("\tf_location_spike_cache : %u\n", wram_block_bytes.f_location_spike_cache);
        printf("\tl_location_spike_cache : %u\n", wram_block_bytes.l_location_spike_cache);
        printf("\tl_feature_spike_cache  : %u\n", wram_block_bytes.l_feature_spike_cache);
        printf("\to_internal_spike_cache : %u\n", wram_block_bytes.o_internal_spike_cache);
        printf("\to_external_spike_cache : %u\n", wram_block_bytes.o_external_spike_cache);
#endif 

#define MAP_CONTEXT_OUTER_STRIDE(outer_strides, wram_block_bytes, num_tasklets, num_cells, part) \
    outer_strides.part = num_tasklets * wram_block_bytes.part * num_cells

#define MAP_SPIKE_COUNT_CACHE_OUTER_STRIDE(outer_strides, wram_block_bytes, num_tasklets, part) \
    outer_strides.part = num_tasklets * wram_block_bytes.part

        MAP_CONTEXT_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, p.location.cells, l_location_context);
        MAP_CONTEXT_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, p.location.cells, l_feature_context);

        MAP_CONTEXT_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, p.features.cells, f_feature_context);
        MAP_CONTEXT_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, p.features.cells, f_location_context);

        MAP_CONTEXT_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, 1, o_internal_context);
        MAP_CONTEXT_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, 1, o_external_context);
        MAP_CONTEXT_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, 1, o_feedforward);

        MAP_SPIKE_COUNT_CACHE_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, l_location_spike_cache);
        MAP_SPIKE_COUNT_CACHE_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, l_feature_spike_cache);

        MAP_SPIKE_COUNT_CACHE_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, f_feature_spike_cache);
        MAP_SPIKE_COUNT_CACHE_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, f_location_spike_cache);

        MAP_SPIKE_COUNT_CACHE_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, o_internal_spike_cache);
        MAP_SPIKE_COUNT_CACHE_OUTER_STRIDE(outer_strides, wram_block_bytes, NR_TASKLETS, o_external_spike_cache);
    }

    barrier_wait(&base_initialization_barrier);

    /*
            features            features          ...
            connection           state
               |                   |
        location.predict -> location.activate -> features.predict -> features.activate -> output.predict -> output.activate
    */

    // private working buffers for connection and spike count cache data
    u8* pbuffer1 = wram_alloc_u8(wram_block_bytes.f_feature_context); 
    u8* pbuffer2 = wram_alloc_u8(wram_block_bytes.f_location_context); 

    u8* pbuffer3 = wram_alloc_u8(wram_block_bytes.f_feature_spike_cache);
    u8* pbuffer4 = wram_alloc_u8(wram_block_bytes.f_location_spike_cache);

#define MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, part) \
    base_addresses->part = addresses.part + tasklet_id * wram_block_bytes.part
    
    mram_content_t* base_addresses = &base_addresses_global[tasklet_id]; // not shared by tasklets, each tasklet has its own base address
    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, l_location_context);
    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, l_feature_context);

    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, f_feature_context);
    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, f_location_context);

    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, o_internal_context);
    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, o_external_context);
    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, o_feedforward);

    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, l_location_spike_cache);
    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, l_feature_spike_cache);

    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, f_feature_spike_cache);
    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, f_location_spike_cache);

    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, o_internal_spike_cache);
    MAP_BASE_ADDRESS(base_addresses, addresses, wram_block_bytes, tasklet_id, o_external_spike_cache);

#define RESET_RUNNING_ADDRESS(running_addresses, base_addresses, part) \
    running_addresses->part = base_addresses->part

    // running addresses are set and reset during execution, while addresses and base_addresses stay fixed
    mram_content_t* running_addresses = &running_addresses_global[tasklet_id]; // not shared by tasklets, each tasklet has its own running address

    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, l_location_context);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, l_feature_context);

    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, f_feature_context);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, f_location_context);

    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, o_internal_context);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, o_external_context);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, o_feedforward);

    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, l_location_spike_cache);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, l_feature_spike_cache);

    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, f_feature_spike_cache);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, f_location_spike_cache);

    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, o_internal_spike_cache);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, o_external_spike_cache);

    barrier_wait(&full_initialization_barrier);

#if PRINT == 2
    if (tasklet_id == 0) {      
        printf("DPU WRAM Buffer Addresses:\n");
        printf("\tShared:\n");

        printf("\t\tp: %u\n", (u32) &p);
        printf("\t\tsize_bytes: %u\n", (u32) &size_bytes);
        printf("\t\taddresses: %u\n", (u32) &addresses);

        printf("\t\twram_block_bytes: %u\n", (u32) &wram_block_bytes);
        printf("\t\touter_strides: %u\n", (u32) &outer_strides);

        printf("\t\tbase_addresses_global: %u\n", (u32) base_addresses_global);
        printf("\t\trunning_addresses_global: %u\n", (u32) running_addresses_global);

        printf("\t\ttmp_active_prev: %u\n", (u32) tmp_active_prev);

        printf("\t\tf_active: %u\n", (u32) f_active);
        printf("\t\tf_predicted: %u\n",(u32) f_predicted);

        printf("\t\tl_active: %u\n", (u32) l_active);
        printf("\t\tl_predicted: %u\n", (u32) l_predicted);

        printf("\t\to_active: %u\n", (u32) o_active); // PACKED REPRESENTATION
        printf("\t\to_prediction_scores: %u\n", (u32) o_prediction_scores);
        printf("\t\to_prediction_score_counts: %u\n", (u32) o_prediction_score_counts);

        printf("\t\texternal_o_activity: %u\n", (u32) &external_o_activity); 
        printf("\t\texternal_o_activity.data: %u\n", (u32) external_o_activity.data);     

        printf("\t\tinput: %u\n", (u32) input_buffer);

        printf("\tPrivate:\n");
        printf("\t\tbuffer1: %u\n", (u32) pbuffer1);
        printf("\t\tbuffer2: %u\n", (u32) pbuffer2);

        printf("\t\tbuffer3: %u\n", (u32) pbuffer3);
        printf("\t\tbuffer4: %u\n", (u32) pbuffer4);
    }
#endif

#if DPU_CHECK_CONNS == 1
    u32 conns_are_healthy = 1;
    for(u32 col = tasklet_id; col < p.location.cols; col += NR_TASKLETS) {
        for(u32 cell = 0; cell < p.location.cells; ++cell) {

            /** Connection data per cell is 1680 B (segments * sizeof(segment_t))
                We mram_read the 2 * 840 B of data into the wram buffer before using it */
            MRAM_READ_COMPONENT(pbuffer1, l_location_context);
            MRAM_READ_COMPONENT(pbuffer2, l_feature_context);

            conns_are_healthy &= htm_connections_check(
                (segment_t*) pbuffer1,
                LOCATION_INDEX_TYPE, 6,
                20, 40, // min_conns, max_conns
                1023, 7, // col_max, cell_max
                4, // lm_max
                1,
                col, cell, tasklet_id
            );

            conns_are_healthy &= htm_connections_check(
                (segment_t*) pbuffer2,
                FEATURE_INDEX_TYPE, 6,
                20, 40, // min_conns, max_conns
                1023, 7, // col_max, cell_max
                4, // lm_max
                1,
                col, cell, tasklet_id
            );

            running_addresses->l_location_context += wram_block_bytes.l_location_context;
            running_addresses->l_feature_context += wram_block_bytes.l_feature_context;
        }

        running_addresses->l_location_context += outer_strides.l_location_context;
        running_addresses->l_feature_context += outer_strides.l_feature_context;
    }

    printf("!!!connections health: %u\n", conns_are_healthy);
#endif

    // ==================== LOCATION LAYER PREDICT =====================

    // Parallelization over columns [col 0 -> t0, col 1 -> t1, ..., col10 -> t10, col 11 -> t0, ...]
    for(u32 col = tasklet_id; col < p.location.cols; col += NR_TASKLETS) {
        u32 pred_bitarray = 0; // 000...000

        /** Spike count cache per col is 48 B (cells * segments * sizeof(segment_t))
                We mram_read the 2 * 48 B of data into the wram buffer before using it */
        MRAM_READ_COMPONENT(pbuffer3, l_location_spike_cache);
        MRAM_READ_COMPONENT(pbuffer4, l_feature_spike_cache);

        for(u32 cell = 0; cell < p.location.cells; ++cell) {
            u32 num_spiking_segments = 0;

            /** Connection data per cell is 1680 B (segments * sizeof(segment_t))
                We mram_read the 2 * 840 B of data into the wram buffer before using it */
            MRAM_READ_COMPONENT(pbuffer1, l_location_context);

            htm_prediction_integrate_context_LOCATION(&num_spiking_segments,
                (segment_t*) pbuffer1, pbuffer3,
                l_active, NULL,
                p.location.location_segments, 
                p.location.htm.permanence_threshold, p.location.htm.segment_spiking_threshold
            );
            
            MRAM_READ_COMPONENT(pbuffer2, l_feature_context);

            htm_prediction_integrate_context_FEATURE(&num_spiking_segments,
                (segment_t*) pbuffer2, pbuffer4,
                f_active, NULL,
                p.location.feature_segments, 
                p.location.htm.permanence_threshold, p.location.htm.segment_spiking_threshold
            );

            // cell is predicted if there is at least one spiking segment
            u32 cell_is_predicted = num_spiking_segments >= 1;
            pred_bitarray |= (cell_is_predicted << cell);

            running_addresses->l_location_context += wram_block_bytes.l_location_context;
            running_addresses->l_feature_context += wram_block_bytes.l_feature_context;
        }
        
        l_predicted[col] = pred_bitarray;

        MRAM_WRITE_COMPONENT(pbuffer3, l_location_spike_cache);
        MRAM_WRITE_COMPONENT(pbuffer4, l_feature_spike_cache);

        running_addresses->l_location_context += outer_strides.l_location_context;
        running_addresses->l_feature_context += outer_strides.l_feature_context;

        running_addresses->l_location_spike_cache += outer_strides.l_location_spike_cache;
        running_addresses->l_feature_spike_cache += outer_strides.l_feature_spike_cache;
    }

    // ==================== LOCATION LAYER ACTIVATE =====================

    // 0. Reset & save state
    for(u32 col = tasklet_id; col < p.location.cols; col += NR_TASKLETS) {
        tmp_active_prev[col] = l_active[col]; // remember the active cols here
        l_active[col] = 0; // set all columns to be inactive, active columns will be selectively activated later in this function
    }

    barrier_wait(&location_act_reset_end_barrier); // need to wait for l_active_prev to settle fully
    // the above barrier also makes sure l_predicted is settled before we use it right down here
    
    // 1. movement -> active columns i.e. shift the grid cell activity with movement
    for(u32 col = tasklet_id; col < p.location.cols; col += NR_TASKLETS) {
        // if the column was active, we need to shift this activity through movement, and compute the next column that should be active
        if(tmp_active_prev[col]) {
            u32 cols_sqrt = 1 << p.location.log_cols_sqrt; 

            u32 x = col % cols_sqrt;
            u32 y = col >> p.location.log_cols_sqrt; // col_it / cols_sqrt

            u32 new_x = (x + (movement_x >> p.location.log_scale.x)) % cols_sqrt; // grid cells wrap around
            u32 new_y = (y + (movement_y >> p.location.log_scale.y)) % cols_sqrt; // grid cells wrap around

            u32 new_col = new_x + (new_y << p.location.log_cols_sqrt);

            l_active[new_col] = 1;
        }
    }

    barrier_wait(&location_col_act_end_barrier); // need to wait for l_active to settle fully

    // 2. predicted cells & active columns -> active cells
    for(u32 col = tasklet_id; col < p.location.cols; col += NR_TASKLETS) {
        if(l_active[col] != 0) {
            u32 act_bitarray = 0; // 000...000 
            for(u32 cell = 0; cell < p.location.cells; ++cell) {
                u32 was_predicted = GET_BIT(l_predicted[col], cell);
                act_bitarray |= (was_predicted << cell);
            }
            // if no cell was activated through predictions, activate all cells in column
            if(act_bitarray == 0) act_bitarray = ~ ((u32) 0); // 111...111

            l_active[col] = act_bitarray;
        } else {
            l_active[col] = 0; // 000...0000
        }
    } 

    barrier_wait(&location_cell_act_end_barrier); // need to wait for l_active to settle fully
    // since l_active is reused right down here

    // 3. learning (involves traversing parts of the segments data)
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, l_location_context);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, l_feature_context);

    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, l_location_spike_cache);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, l_feature_spike_cache);

    for(u32 col = tasklet_id; col < p.location.cols; col += NR_TASKLETS) {

        MRAM_READ_COMPONENT(pbuffer3, l_location_spike_cache);
        MRAM_READ_COMPONENT(pbuffer4, l_feature_spike_cache);

        /**
         * If the columns is actived but no cell was predicted, we have to select the "winning cell"
         * to that end, we iterate through all the cells and we find the cell with the segment that was closest to becoming active
         */
        u32 winning_cell = 0;
        u32 winning_connection_count = 0;
        u32 col_active_and_unpredicted = l_active[col] != 0 && l_predicted[col] == 0;
        if(col_active_and_unpredicted) {
            // We find the cell with the segment that had most activity
            htm_learning_pick_winner_cell(
                &winning_cell, &winning_connection_count, 
                pbuffer3, 
                p.location.location_segments, p.location.cells
            );

            htm_learning_pick_winner_cell(
                &winning_cell, &winning_connection_count, 
                pbuffer4, 
                p.location.feature_segments, p.location.cells
            );
        }

        for(u32 cell = 0; cell < p.location.cells; ++cell) {

            u32 cell_is_active = GET_BIT(l_active[col], cell);
            u32 cell_is_predicted = GET_BIT(l_predicted[col], cell);

            MRAM_READ_COMPONENT(pbuffer1, l_location_context);
            i32 modified = htm_learning_adjust_permanences(cell,
                (segment_t*) pbuffer1, pbuffer3,
                tmp_active_prev, NULL,
                cell_is_active, cell_is_predicted,
                col_active_and_unpredicted, winning_cell,
                LOCATION_INDEX_TYPE, p.location.location_segments,
                &p.location.htm, 1 // enable decay
            );

            if(modified) {
                MRAM_WRITE_COMPONENT(pbuffer1, l_location_context);
            }

            MRAM_READ_COMPONENT(pbuffer2, l_feature_context);
            modified = htm_learning_adjust_permanences(cell,
                (segment_t*) pbuffer2, pbuffer4,
                f_active, NULL,
                cell_is_active, cell_is_predicted,
                col_active_and_unpredicted, winning_cell,
                FEATURE_INDEX_TYPE, p.location.feature_segments,
                &p.location.htm, 1 // enable decay
            );

            if(modified) {
                MRAM_WRITE_COMPONENT(pbuffer2, l_feature_context);
            }

            running_addresses->l_location_context += wram_block_bytes.l_location_context;
            running_addresses->l_feature_context += wram_block_bytes.l_feature_context;
        } 
        
        running_addresses->l_location_context += outer_strides.l_location_context;
        running_addresses->l_feature_context += outer_strides.l_feature_context;

        running_addresses->l_location_spike_cache += outer_strides.l_location_spike_cache;
        running_addresses->l_feature_spike_cache += outer_strides.l_feature_spike_cache;
    }

    // ==================== FEATURE LAYER PREDICT =====================

    // no need for a barrier after the location learning step since l_active was already settled above it

    // Parallelization over columns [col 0 -> t0, col 1 -> t1, ..., col10 -> t10, col 11 -> t0, ...]
    for(u32 col = tasklet_id; col < p.features.cols; col += NR_TASKLETS) {
        u32 pred_bitarray = 0; // 000...000

        MRAM_READ_COMPONENT(pbuffer3, f_feature_spike_cache);
        MRAM_READ_COMPONENT(pbuffer4, f_location_spike_cache);

        for(u32 cell = 0; cell < p.features.cells; ++cell) {
            u32 num_spiking_segments = 0;

            MRAM_READ_COMPONENT(pbuffer1, f_feature_context);

            htm_prediction_integrate_context_FEATURE(&num_spiking_segments,
                (segment_t*) pbuffer1, pbuffer3,
                f_active, NULL, 
                p.features.feature_segments, 
                p.features.htm.permanence_threshold, p.features.htm.segment_spiking_threshold
            );

            MRAM_READ_COMPONENT(pbuffer2, f_location_context);

            htm_prediction_integrate_context_LOCATION(&num_spiking_segments,
                (segment_t*) pbuffer2, pbuffer4,
                l_active, NULL,
                p.features.location_segments, 
                p.features.htm.permanence_threshold, p.features.htm.segment_spiking_threshold
            );

            // cell is predicted if there is at least one spiking segment
            u32 cell_is_predicted = num_spiking_segments >= 1;
            pred_bitarray |= (cell_is_predicted << cell);

            running_addresses->f_feature_context += wram_block_bytes.f_feature_context;
            running_addresses->f_location_context += wram_block_bytes.f_location_context;
        }
        
        f_predicted[col] = pred_bitarray;

        MRAM_WRITE_COMPONENT(pbuffer3, f_feature_spike_cache);
        MRAM_WRITE_COMPONENT(pbuffer4, f_location_spike_cache);

        running_addresses->f_feature_context += outer_strides.f_feature_context;
        running_addresses->f_location_context += outer_strides.f_location_context;

        running_addresses->f_feature_spike_cache += outer_strides.f_feature_spike_cache;
        running_addresses->f_location_spike_cache += outer_strides.f_location_spike_cache;
    }

    barrier_wait(&feature_predict_end_barrier); // need for f_predicted to settle

    // ==================== FEATURE LAYER ACTIVATE =====================

    // 1. predicted cells & active columns -> active cells
    for(u32 col = tasklet_id; col < p.features.cols; col += NR_TASKLETS) {
        tmp_active_prev[col] = f_active[col]; // remember the active cols here

        if(features[col] != 0) {
            u32 act_bitarray = 0; // 000...000 
            for(u32 cell = 0; cell < p.features.cells; ++cell) {
                u32 was_predicted = GET_BIT(f_predicted[col], cell);
                act_bitarray |= (was_predicted << cell);
            }
            // if no cell was activated through predictions, activate all cells in column
            if(act_bitarray == 0) act_bitarray = ~ ((u32) 0); // 111...111

            f_active[col] = act_bitarray;
        } else {
            f_active[col] = 0; // 000...0000
        }
    } 

    barrier_wait(&feature_cell_act_end_barrier); // need for f_active to settle

    // 2. learning (involves traversing the segments data)
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, f_feature_context);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, f_location_context);

    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, f_feature_spike_cache);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, f_location_spike_cache);

    for(u32 col = tasklet_id; col < p.features.cols; col += NR_TASKLETS) {
        
        MRAM_READ_COMPONENT(pbuffer3, f_feature_spike_cache);
        MRAM_READ_COMPONENT(pbuffer4, f_location_spike_cache);

        /**
         * If the columns is actived but no cell was predicted, we have to select the "winning cell"
         * to that end, we iterate through all the cells and we find the cell with the segment that was closest to becoming active
         */
        u32 winning_cell = 0;
        u32 winning_connection_count = 0;
        u32 col_active_and_unpredicted = f_active[col] != 0 && f_predicted[col] == 0;
        if(col_active_and_unpredicted) {
            // We find the cell with the segment that had most activity
            htm_learning_pick_winner_cell(
                &winning_cell, &winning_connection_count, 
                pbuffer3, p.features.feature_segments, p.features.cells
            );

            htm_learning_pick_winner_cell(
                &winning_cell, &winning_connection_count, 
                pbuffer4, p.features.location_segments, p.features.cells
            );
        }

        for(u32 cell = 0; cell < p.features.cells; ++cell) {

            u32 cell_is_active = GET_BIT(f_active[col], cell);
            u32 cell_is_predicted = GET_BIT(f_predicted[col], cell);

            MRAM_READ_COMPONENT(pbuffer1, f_feature_context);

            i32 modified = htm_learning_adjust_permanences(cell,
                (segment_t*) pbuffer1, pbuffer3,
                tmp_active_prev, NULL,
                cell_is_active, cell_is_predicted,
                col_active_and_unpredicted, winning_cell,
                FEATURE_INDEX_TYPE, p.features.feature_segments,
                &p.features.htm, 1 // enable decay
            );

            if(modified) {
                MRAM_WRITE_COMPONENT(pbuffer1, f_feature_context);
            }

            MRAM_READ_COMPONENT(pbuffer1, f_location_context);

            modified = htm_learning_adjust_permanences(cell,
                (segment_t*) pbuffer2, pbuffer4,
                l_active, NULL,
                cell_is_active, cell_is_predicted,
                col_active_and_unpredicted, winning_cell,
                LOCATION_INDEX_TYPE, p.features.location_segments,
                &p.features.htm, 1 // enable decay
            );

            if(modified) {
                MRAM_WRITE_COMPONENT(pbuffer2, f_location_context);
            }

            running_addresses->f_feature_context += wram_block_bytes.f_feature_context;
            running_addresses->f_location_context += wram_block_bytes.f_location_context;
        }  

        running_addresses->f_feature_context += outer_strides.f_feature_context;
        running_addresses->f_location_context += outer_strides.f_location_context;

        running_addresses->f_feature_spike_cache += outer_strides.f_feature_spike_cache;
        running_addresses->f_location_spike_cache += outer_strides.f_location_spike_cache;
    }

    // ==================== OUTPUT LAYER PREDICT =====================

    u32 running_num_cells = 0;

    for(u32 cell = tasklet_id; cell < p.output.cells; cell += NR_TASKLETS) {
        // maybe get the new spike cache block at the beginning of the iteration
        if(running_num_cells % O_SPIKE_CACHE_BLOCK_NUM_CELLS == 0) {
            MRAM_READ_COMPONENT(pbuffer3, o_internal_spike_cache);
            MRAM_READ_COMPONENT(pbuffer4, o_external_spike_cache);
        }
        running_num_cells += 1;

        u32 num_spiking_segments = 0;

        MRAM_READ_COMPONENT(pbuffer1, o_internal_context);

        htm_prediction_integrate_context_INTERNAL_OUTPUT(&num_spiking_segments,
            (segment_t*) pbuffer1, pbuffer3,
            o_active, NULL, 
            p.output.internal_context_segments, 
            p.output.htm.permanence_threshold, p.output.htm.segment_spiking_threshold
        );

        MRAM_READ_COMPONENT(pbuffer2, o_external_context);

        htm_prediction_integrate_context_EXTERNAL_OUTPUT(&num_spiking_segments,
            (segment_t*) pbuffer2, pbuffer4,
            NULL, &external_o_activity,
            p.output.external_context_segments, 
            p.output.htm.permanence_threshold, p.output.htm.segment_spiking_threshold
        );

        if(num_spiking_segments > 255) num_spiking_segments = 255;

        o_prediction_scores[cell] = num_spiking_segments;

        running_addresses->o_internal_context += outer_strides.o_internal_context;
        running_addresses->o_external_context += outer_strides.o_external_context;

        if(running_num_cells % O_SPIKE_CACHE_BLOCK_NUM_CELLS == 0) {
            // maybe commit modified spike cache at the end of the iteration
            MRAM_WRITE_COMPONENT(pbuffer3, o_internal_spike_cache);
            MRAM_WRITE_COMPONENT(pbuffer4, o_external_spike_cache);

            // go to the next spike cache block address
            running_addresses->o_internal_spike_cache += outer_strides.o_internal_spike_cache;
            running_addresses->o_external_spike_cache += outer_strides.o_external_spike_cache;
        }
    }
    
    // ==================== OUTPUT LAYER ACTIVATE =====================

    // reset frequency counts
    for(u32 i = 0; i < p.output.internal_context_segments + p.output.external_context_segments; ++i) 
        o_prediction_score_counts[i] = 0;

    for(u32 i = tasklet_id; i < p.output.cells >> 5; i += NR_TASKLETS)
        tmp_active_prev[i] = o_active[i];

    barrier_wait(&output_predict_end_barrier); // wait for o_prediction_scores to settle

    // 1. collect the counts of the cells with top k best context support
    for(u32 cell = tasklet_id; cell < p.output.cells; cell += NR_TASKLETS) {
        u8 score = o_prediction_scores[cell];
        o_prediction_score_counts[score] += 1;
    }

    barrier_wait(&output_collect_counts_end_barrier); // wait for o_prediction_score_counts to settle

    // get the exact value of the cell with k-th best context support
    u16 kth_largest = find_kth_largest_from_score_counts(
        o_prediction_score_counts, 
        p.output.internal_context_segments + p.output.external_context_segments - 1, 
        p.output.extended_htm.min_active_cells
    );

    barrier_wait(&output_find_kth_end_barrier); // wait for kth_largest to settle

    // 2. determine the active cells, those that have enough ffw support 
    //      AND that have more ctx support than the k-th cell with most ctx support (from step 1)

    for(u32 cell = tasklet_id; cell < p.output.cells; cell += NR_TASKLETS) {
        u32 cell_response = 0;

        MRAM_READ_COMPONENT(pbuffer1, o_feedforward);

        for(u32 conn = 0; conn < ((segment_t*) pbuffer1)->num_connections; ++conn) {
            segment_data seg_data = ((segment_t*) pbuffer1)->connections[conn];

            feature_index index = seg_data.index.feature; 

            u32 cell_is_active = GET_BIT(f_active[index.col], index.cell);
            u32 cell_is_connected = seg_data.permanence >= p.output.extended_htm.feedforward_permanence_threshold;

            cell_response += cell_is_active && cell_is_connected;
        }

        if(
            cell_response >= p.output.extended_htm.feedforward_activation_threshold &&
            o_prediction_scores[cell] >= kth_largest
        ) {
            SET_BIT_IN_PACKED32(o_active, cell);
        } else {
            RESET_BIT_IN_PACKED32(o_active, cell);
        }

        running_addresses->o_feedforward += outer_strides.o_feedforward;
    }

    barrier_wait(&output_cell_act_end_barrier); // wait for o_active to settle

    // 3. Learn context connections     
    
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, o_internal_context);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, o_external_context);

    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, o_internal_spike_cache);
    RESET_RUNNING_ADDRESS(running_addresses, base_addresses, o_external_spike_cache);

    running_num_cells = 0;

    MRAM_READ_COMPONENT(pbuffer3, o_internal_spike_cache);
    MRAM_READ_COMPONENT(pbuffer4, o_external_spike_cache);

    for(u32 cell = tasklet_id; cell < p.output.cells; cell += NR_TASKLETS) {
        running_num_cells += 1;

        u32 cell_is_active = GET_BIT_FROM_PACKED32(o_active, cell);
        u32 cell_is_predicted = o_prediction_scores[cell] >= 1;

        MRAM_READ_COMPONENT(pbuffer1, o_internal_context);

        i32 modified = htm_learning_adjust_permanences(cell, 
            (segment_t*) pbuffer1, pbuffer3,
            tmp_active_prev, NULL,
            cell_is_active, cell_is_predicted,
            0, 0, // no column, no winning cell
            INTERNAL_OUTPUT_INDEX_TYPE, p.output.internal_context_segments,
            &p.output.htm, 0 // no decay
        );

        if(modified) {
            MRAM_WRITE_COMPONENT(pbuffer1, o_internal_context);
        }

        MRAM_READ_COMPONENT(pbuffer2, o_external_context);

        modified = htm_learning_adjust_permanences(cell, 
            (segment_t*) pbuffer2, pbuffer4,
            NULL, &external_o_activity,
            cell_is_active, cell_is_predicted,
            0, 0, // no column, no winning cell
            EXTERNAL_OUTPUT_INDEX_TYPE, p.output.external_context_segments,
            &p.output.htm, 0 // no decay
        );

        if(modified) {
            MRAM_WRITE_COMPONENT(pbuffer2, o_external_context);
        }

        running_addresses->o_internal_context += outer_strides.o_internal_context;
        running_addresses->o_external_context += outer_strides.o_external_context;

        if(running_num_cells % O_SPIKE_CACHE_BLOCK_NUM_CELLS == 0) {
            // commit modified spike cache
            MRAM_WRITE_COMPONENT(pbuffer3, o_internal_spike_cache);
            MRAM_WRITE_COMPONENT(pbuffer4, o_external_spike_cache);

            // go to the next spike cache block address
            running_addresses->o_internal_spike_cache += outer_strides.o_internal_spike_cache;
            running_addresses->o_external_spike_cache += outer_strides.o_external_spike_cache;

            // get the new spike cache block
            MRAM_READ_COMPONENT(pbuffer3, o_internal_spike_cache);
            MRAM_READ_COMPONENT(pbuffer4, o_external_spike_cache);
        }
    }

    if(tasklet_id == 0) {
        mram_write(
            o_active,
            (__mram_ptr void*) addresses.output,
            size_bytes.output
        );

#if PRINT == 2
        printf("Sparsities:\n");

        printf("\t l_pred ");
        print_packed_spvec_u32(l_predicted, p.location.cols);
        printf("\n\t l_act ");
        print_packed_spvec_u32(l_active, p.location.cols);

        printf("\t\t f_pred ");
        print_packed_spvec_u32(f_predicted, p.features.cols);
        printf("\n\t f_act ");
        print_packed_spvec_u32(f_active, p.features.cols);

        printf("\t\t o_act ");
        print_packed_spvec_u32(o_active, p.output.cells);
        printf("\n\t\t ext_o ");
        print_packed_spvec_u32(external_o_activity.data, p.output.external_cells >> 5);
        printf("\n\t\t o_pred_score_counts ");
        for(u32 score = 0; score < p.output.internal_context_segments + p.output.external_context_segments; ++score) {
            printf("%u: %u, ", score, o_prediction_score_counts[score]);
        }
        printf("\n");

        printf("\t features ");
        print_spvec_u8(features, p.features.cols);
        printf("\n\t movement ");
        printf("[%d, %d]", movement_x, movement_y);
        printf("\n");
#endif

    }


// #if defined(CYCLES) || defined(INSTRUCTIONS)
//     result->count += perfcounter_get(); // STOP TIMER
// #endif
	
    return 0;
}

#include "tinylib.c"
