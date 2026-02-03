/**
* app.c
* Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"

#include "../tbtc-htm/tensor.h"
#include "../tbtc-htm/grid_environment.h"
#include "../tbtc-htm/sensor_module.h"
#include "../tbtc-htm/learning_module.h"
#include "../tbtc-htm/motor_policy.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

#ifndef NR_DPUS
#define NR_DPUS 1
#endif

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#ifndef NUM_EXTERNAL_LMS
#define NUM_EXTERNAL_LMS 5
#endif

#ifndef OUT_CELL_LOG_DIM
#define OUT_CELL_LOG_DIM 10 
#endif

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0') 

#define NUM_SAMPLES(num_dpus, num_samples, dpu_idx) (num_samples / num_dpus + (num_samples % num_dpus > dpu_idx ? 1 : 0))

#define RETRIEVE_RESULTS(field) \
    do { \
        dpu_it = 0; \
        DPU_FOREACH(dpu_set, dpu, dpu_it) { \
            results[dpu_it].field = 0; \
            for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) { \
                if (results_retrieve[dpu_it][each_tasklet].field > results[dpu_it].field) \
                    results[dpu_it].field = results_retrieve[dpu_it][each_tasklet].field; \
            } \
        } \
        u64 max_count_##field = 0; \
        u64 min_count_##field = 0xFFFFFFFFFFFFFFFF; \
        dpu_it = 0; \
        DPU_FOREACH(dpu_set, dpu) { \
            if(results[dpu_it].field > max_count_##field) \
                max_count_##field = results[dpu_it].field; \
            if(results[dpu_it].field < min_count_##field) \
                min_count_##field = results[dpu_it].field; \
            dpu_it++; \
        } \
        cc_##field += (double) max_count_##field; \
        cc_##field /= (double) NR_TASKLETS; \
    } while(0)

void transfer_parameters_to_dpus(
    struct dpu_set_t dpu_set,
    mram_content_t* mram_size_bytes, 
    mram_content_t* mram_addresses, 
    dpu_model_params_t* model_params
) {
    u32 dpu_it = 0;
    struct dpu_set_t dpu;

    DPU_FOREACH(dpu_set, dpu, dpu_it) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, model_params));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "p", 0, sizeof(*model_params), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, dpu_it) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, mram_size_bytes));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "size_bytes", 0, sizeof(*mram_size_bytes), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, dpu_it) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, mram_addresses));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "addresses", 0, sizeof(*mram_addresses), DPU_XFER_DEFAULT));
}

// we transfer the same model to all dpus, which is not what you want in practice 
// but the underlying computation does not change if we change the model
// also we can discount model transfer in our costs: we only care about input transfer cost
void transfer_model_to_dpus(
    struct dpu_set_t dpu_set,
    mram_content_t mram_addresses,
    mram_content_t mram_size_bytes,
    learning_module* lm
) {
    // =========== CONTEXT CONNECTIONS DATA ================
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.f_feature_context, // offset
        lm->feature_net.in_segments.feature_context, // src
        mram_size_bytes.f_feature_context, // length
        DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.f_location_context, // offset
        lm->feature_net.in_segments.location_context, // src
        mram_size_bytes.f_location_context, // length
        DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.l_location_context, // offset
        lm->location_net.in_segments.location_context, // src
        mram_size_bytes.l_location_context, // length
        DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.l_feature_context, // offset
        lm->location_net.in_segments.feature_context, // src
        mram_size_bytes.l_feature_context, // length
        DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.o_internal_context, // offset
        lm->output_net.in_segments.internal_context, // src
        mram_size_bytes.o_internal_context, // length
        DPU_XFER_DEFAULT));
        DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.o_external_context, // offset
        lm->output_net.in_segments.external_context, // src
        mram_size_bytes.o_external_context, // length
        DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.o_feedforward, // offset
        lm->output_net.in_segments.feedforward, // src
        mram_size_bytes.o_feedforward, // length
        DPU_XFER_DEFAULT));

    // =========== SPIKE COUNT CACHE DATA ================
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.f_feature_spike_cache, // offset
        lm->feature_net.spike_count_cache.feature_segments, // src
        mram_size_bytes.f_feature_spike_cache, // length
        DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.f_location_spike_cache, // offset
        lm->feature_net.spike_count_cache.location_segments, // src
        mram_size_bytes.f_location_spike_cache, // length
        DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.l_location_spike_cache, // offset
        lm->location_net.spike_count_cache.location_segments, // src
        mram_size_bytes.l_feature_spike_cache, // length
        DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.l_feature_spike_cache, // offset
        lm->location_net.spike_count_cache.feature_segments, // src
        mram_size_bytes.l_feature_spike_cache, // length
        DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.o_internal_spike_cache, // offset
        lm->output_net.spike_count_cache.internal_context_segments, // src
        mram_size_bytes.o_internal_spike_cache, // length
        DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.o_external_spike_cache, // offset
        lm->output_net.spike_count_cache.external_context_segments, // src
        mram_size_bytes.o_external_spike_cache, // length
        DPU_XFER_DEFAULT));
}

// we transfer the same input to all dpus, which is not what you want in practice 
// but the underlying computation does not change if we change the input
// input transfer costs matter!
void transfer_inputs_to_dpus(
    struct dpu_set_t dpu_set,
    mram_content_t mram_addresses,
    mram_content_t mram_size_bytes,
    vec2d movement,
    features_t features,
    lmat_u32* external_o_activity
) {
    u32 dpu_it = 0;
    struct dpu_set_t dpu;

    printf("Parallel transfer of movement input (address %u size %u)\n", mram_addresses.input_movement, mram_size_bytes.input_movement);

    DPU_FOREACH(dpu_set, dpu, dpu_it) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &movement)); // src
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, 
        DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.input_movement, // offset
        mram_size_bytes.input_movement, // length
        DPU_XFER_DEFAULT)); 

    printf("Parallel transfer of feature input (address %u size %u)\n", mram_addresses.input_features, mram_size_bytes.input_features);

    DPU_FOREACH(dpu_set, dpu, dpu_it) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, features.active_columns)); // src
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, 
        DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.input_features, // offset
        mram_size_bytes.input_features, // length
        DPU_XFER_DEFAULT)); 

    printf("Parallel transfer of external output activity (address %u size %u)\n", mram_addresses.external_o_activity, mram_size_bytes.external_o_activity);

    DPU_FOREACH(dpu_set, dpu, dpu_it) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, external_o_activity->data)); // src
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, 
        DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.external_o_activity, // offset
        mram_size_bytes.external_o_activity, // length
        DPU_XFER_DEFAULT)); 

    // broadcast_buffer(dpu_set, 
    //     size_bytes.feature_layer + size_bytes.location_layer + size_bytes.output_layer, 
    //     features.active_columns, 
    //     features.num_columns * sizeof(*features.active_columns), "input");
}

void retrieve_output_from_dpus(
    u32* output,
    struct dpu_set_t dpu_set, 
    mram_content_t mram_addresses,
    mram_content_t mram_size_bytes
) {
    u32 dpu_it = 0;
    struct dpu_set_t dpu;

    printf("Pulling output \n");

    DPU_FOREACH(dpu_set, dpu, dpu_it) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, output));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, 
        DPU_MRAM_HEAP_POINTER_NAME, 
        mram_addresses.output, 
        mram_size_bytes.output, 
        DPU_XFER_DEFAULT));
}

// Main of the Host Application
int main(int argc, char **argv) {

    // Input parameters
    struct Params p = input_params(argc, argv);

    // Timer declaration
    Timer timer;
#if defined(CYCLES) || defined(INSTRUCTIONS)
    double cc_count = 0;
#endif

    u32 num_step = 10;

    printf("Generating environment\n");

    // srand(time(NULL));

    grid_t env;
    u32 env_sidelen = 10;
    init_grid_env(&env, env_sidelen, env_sidelen);
    populate_grid_env_random(&env);

    grid_t patch; // grid_t is an ugly abstraction for the patch
    u32 patch_sidelen = 3;
    init_grid_env(&patch, patch_sidelen, patch_sidelen);
    uvec2d patch_center = (uvec2d) { .x = patch_sidelen / 2, .y = patch_sidelen / 2 };

    bounds_t bounds = get_bounds(env_sidelen, env_sidelen, patch_sidelen, patch_sidelen);

    uvec2d agent_location = { .x = 5, .y = 1 }; // start location

    u32 num_cols = 1024;

    lmat_u32 external_o_activity;
    lmat_u32_init(&external_o_activity, NUM_EXTERNAL_LMS, OUT_CELL_LOG_DIM - 5); // packed on u32 = 2^5 bits

    // generate external layer activity with around 2-3% sparsity
    for(u32 lm = 0; lm < NUM_EXTERNAL_LMS; ++lm) {
        for(u32 i = 0; i < 1 << (OUT_CELL_LOG_DIM - 5); ++ i) {
            u32 non_null_bit = unif_rand_range_u32(0, 31);
            *LMATP(external_o_activity, lm, i) = 1 << non_null_bit;
        }
    }

    grid_sm sm;
    init_sensor_module(&sm, GRID_ENV_MIN_VALUE, GRID_ENV_MAX_VALUE, num_cols);

#if PRINT
    print_pooler(&sm.pooler);
#endif

    random_motor_policy_t motor_policy;
    init_random_motor_policy(&motor_policy, agent_location, bounds, num_step);

    features_t f;
    init_features(&f, sm.pooler.params.num_minicols, sm.pooler.params.top_k);

    printf("--- step %u: agent at location (%u, %u)\n", 0, agent_location.x, agent_location.y);
    extract_patch(&patch, &env, agent_location, patch_sidelen);

    print_grid(&patch);

    sensor_module(sm, &f, patch, patch_center);

    printf("column activation sparsity: ");
    print_spvec_u8(f.active_columns, f.num_columns);

    printf("\nGenerating model\n");

    htm_params_t htm_params = (htm_params_t) {
        .permanence_threshold = REPR_u8(0.5),
        .segment_spiking_threshold = 15,

        .perm_increment = REPR_u8(0.06f),
        .perm_decrement = REPR_u8(0.04f),
        .perm_decay = 1 // 1/256, the smallest possible non-zero decay
    };

    extended_htm_params_t ext_htm_params = (extended_htm_params_t) {
        .feedforward_permanence_threshold = REPR_u8(0.5),
        .context_permanence_threshold = REPR_u8(0.5),

        .feedforward_activation_threshold = 3,
        .context_activation_threshold = 18,

        .min_active_cells = 10,
    };

    output_layer_params_t output_p = (output_layer_params_t) {
        .cells = 1 << OUT_CELL_LOG_DIM, // cells per col
        .log_cells = OUT_CELL_LOG_DIM,

        .internal_context_segments = 5,
        .external_context_segments = 5,

        .external_cells = 1 << OUT_CELL_LOG_DIM, // external output dim
        .log_external_cells = OUT_CELL_LOG_DIM, // external output dim
        .external_lms = NUM_EXTERNAL_LMS, // num of connected external lms
        
        .htm = htm_params,
        .extended_htm = ext_htm_params
    };

    feature_layer_params_t features_p = (feature_layer_params_t) {
        .cols = num_cols,
        .cells = 8, // cells per col

        .feature_segments = 5,
        .location_segments = 5,

        .htm = htm_params
    };

    location_layer_params_t location_p = (location_layer_params_t) {
        .cols = num_cols,
        .log_cols_sqrt = (u32) log2(sqrt(num_cols)), // 5
        .cells = 8,

        .location_segments = 5,
        .feature_segments = 5,

        .log_scale = (uvec2d) { .x = 0, .y = 0 },

        .htm = htm_params
    };
 
    learning_module lm;
    init_learning_module(
        &lm, 
        output_p, 
        features_p,
        location_p
    );

    vec2d movement = { .x = 0, .y = 0 };

#if PRINT
    print_grid(&env);
#endif

    printf("Load DPU arguments\n");
    dpu_model_params_t model_params = (dpu_model_params_t) {
        .output = output_p,
        .features = features_p,
        .location = location_p
    };
    mram_content_t mram_size_bytes = (mram_content_t) {
        .f_feature_context = feature_layer_get_feature_context_footprint_bytes(features_p), // 8B aligned via sizeof(segment_t)
        .f_location_context = feature_layer_get_location_context_footprint_bytes(features_p), // 8B aligned via sizeof(segment_t)

        .l_location_context = location_layer_get_location_context_footprint_bytes(location_p), // 8B aligned via sizeof(segment_t)
        .l_feature_context = location_layer_get_feature_context_footprint_bytes(location_p), // 8B aligned via sizeof(segment_t)

        .o_internal_context = output_layer_get_internal_context_footprint_bytes(output_p), // 8B aligned via sizeof(segment_t)
        .o_external_context = output_layer_get_external_context_footprint_bytes(output_p), // 8B aligned via sizeof(segment_t)
        .o_feedforward = output_layer_get_feedforward_footprint_bytes(output_p), // 8B aligned via sizeof(segment_t)

        .input_movement = sizeof(movement), // 8B
        .input_features = ROUND_UP_TO_MULTIPLE_OF_8(f.num_columns * sizeof(*f.active_columns)), 

        .external_o_activity = lmat_u32_count(&external_o_activity) * sizeof(*external_o_activity.data),

        .output = (output_p.cells >> 5) * sizeof(u32), // 128 B

        .f_feature_spike_cache = feature_layer_get_feature_segments_spike_count_cache_bytes(features_p),
        .f_location_spike_cache = feature_layer_get_location_segments_spike_count_cache_bytes(features_p),

        .l_location_spike_cache = location_layer_get_location_segments_spike_count_cache_bytes(location_p),
        .l_feature_spike_cache = location_layer_get_feature_segments_spike_count_cache_bytes(location_p),

        .o_internal_spike_cache = output_layer_get_internal_context_segments_spike_count_cache_bytes(output_p),
        .o_external_spike_cache = output_layer_get_external_context_segments_spike_count_cache_bytes(output_p),
    };

#define MAP_REGION(addresses, size_bytes, field_name, mapping_offset) do { \
    addresses.field_name = mapping_offset; \
    mapping_offset += size_bytes.field_name; \
} while(0)

    mram_content_t mram_addresses;
    u32 offset = 0;

    MAP_REGION(mram_addresses, mram_size_bytes, f_feature_context, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, f_location_context, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, l_location_context, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, l_feature_context, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, o_internal_context, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, o_external_context, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, o_feedforward, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, input_movement, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, input_features, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, external_o_activity, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, output, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, f_feature_spike_cache, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, f_location_spike_cache, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, l_location_spike_cache, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, l_feature_spike_cache, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, o_internal_spike_cache, offset);
    MAP_REGION(mram_addresses, mram_size_bytes, o_external_spike_cache, offset);

    PRINT_MODEL_PARAMS(model_params);
    PRINT_MRAM_CONTENT(mram_size_bytes, mram_addresses);

    u32 num_dpus_total = 1;
    // u32 num_dpus_total = 2560;
    if(num_dpus_total == 0) {
        printf("WARNING: Running in 1 DPU test mode.\n");
    }

    struct dpu_set_t dpu_set, dpu;
    u32 effective_num_dpus_total;
    // DPU_ASSERT(dpu_alloc(num_dpus_total, "sgXferEnable=true, regionMode=safe, enableProfiling=sections", &dpu_set));
    DPU_ASSERT(dpu_alloc(num_dpus_total, NULL, &dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &effective_num_dpus_total)); // Number of DPUs in the DPU set
    if(effective_num_dpus_total != num_dpus_total) {
        printf("ERROR: Allocated %d DPU(s) (Requested %d)\t", effective_num_dpus_total, num_dpus_total);
        return 0;
    } else {
        printf("Allocated %d DPU(s), Each Running %d Tasklets\n", num_dpus_total, NR_TASKLETS);
    }

    // Load binary
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    // Transfer model to DPUs (does not count in the CPU-DPU transfer times as in practice it could be hidden
    //      by preloading models in memory, same for these "arguments")

    transfer_parameters_to_dpus(dpu_set, &mram_size_bytes, &mram_addresses, &model_params);
    transfer_model_to_dpus(dpu_set, mram_addresses, mram_size_bytes, &lm);

    start(&timer, 0, 0); // Start timer (CPU-DPU transfers)

    transfer_inputs_to_dpus(dpu_set, mram_addresses, mram_size_bytes, movement, f, &external_o_activity);

    stop(&timer, 0); // Stop timer (CPU-DPU transfers)

    // Run on DPUs

    printf("Run on DPUs\n");

    start(&timer, 1, 0); // Start timer (DPU kernel)

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    stop(&timer, 1); // Stop timer (DPU kernel)

#if PRINT == 2
    {
        u32 each_dpu = 0;
        printf("Display DPU Logs\n");
        DPU_FOREACH (dpu_set, dpu) {
            printf("DPU#%d:\n", each_dpu);
            DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
            each_dpu++;
        }
    }
#endif

    // Retrieve results
    printf("Retrieve results\n");
    u32* output;
    output = calloc(output_p.cells >> 5, sizeof(u32));
    start(&timer, 2, 0); // Start timer (DPU-CPU transfers)

    retrieve_output_from_dpus(
        output, 
        dpu_set,
        mram_addresses,
        mram_size_bytes
    );

    stop(&timer, 2); // Stop timer (DPU-CPU transfers)

    printf("Output: ");
    print_packed_spvec_u32(output, lm.output_net.p.cells);
    printf("\n");

    // Benchmark the host
    printf("Run on host\n");
    start(&timer, 3, 0);
    
    learning_module_step(&lm, f, movement, &external_o_activity);
    print_packed_spvec_u32(lm.output_net.active, lm.output_net.p.cells);
    printf("\n");

    stop(&timer, 3);

    u32 equal_output = 1;
    for(u32 word = 0; word < lm.output_net.p.cells >> 5; ++word) {
        if(output[word] != lm.output_net.active[word]) 
            equal_output = 0;
    }

    if(equal_output)
        printf("dpu and host results are equal\n");
    else
        printf("!dpu and host results are NOT equal!\n");


#ifdef CSV
    printf("results_and_timings(cycles), %d, %d", num_dpus_total, NR_TASKLETS);
#else
    printf("results_and_timings(cycles), %d, %d\n", num_dpus_total, NR_TASKLETS);
#endif

#ifdef CSV
    printf(", ");
    print2(&timer, 0, p.n_reps);
    printf(", ");
    print2(&timer, 1, p.n_reps);
    printf(", ");
    print2(&timer, 2, p.n_reps);
    printf(", ");
    print2(&timer, 3, p.n_reps);
    printf(", ");
    print2(&timer, 4, p.n_reps);
#else 
    printf("CPU-DPU transfers ");
    print(&timer, 0, p.n_reps);
    printf("\nDPU kernel ");
    print(&timer, 1, p.n_reps);
    printf("\nDPU-CPU transfers ");
    print(&timer, 2, p.n_reps);
    printf("\nCPU prediction ");
    print(&timer, 3, p.n_reps);
#endif 
printf("\n");

#if defined(CYCLES) || defined(INSTRUCTIONS)
    dpu_results_t results[num_dpus_total];
    // Parallel transfers
    dpu_results_t* results_retrieve[num_dpus_total];

    dpu_it = 0;
    DPU_FOREACH(dpu_set, dpu, dpu_it) {
        results_retrieve[dpu_it] = (dpu_results_t*)malloc(NR_TASKLETS * sizeof(dpu_results_t));
        DPU_ASSERT(dpu_prepare_xfer(dpu, results_retrieve[dpu_it]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, NR_TASKLETS * sizeof(dpu_results_t), DPU_XFER_DEFAULT));

    RETRIEVE_RESULTS(count);

    dpu_it = 0;
    DPU_FOREACH(dpu_set, dpu, dpu_it) { 
        free(results_retrieve[dpu_it]);
    }
#endif

#ifdef CYCLES
// #ifdef CSV
//     printf("results_and_timings(cycles), %d, %d, %d, %g", nr_of_dpus, NR_TASKLETS, num_samples, cc_count);
// #else
    // retrieve DPU frequency
    u32 clocks_per_sec;
    DPU_FOREACH (dpu_set, dpu) {
        DPU_ASSERT(dpu_copy_from(
            dpu, 
            "CLOCKS_PER_SEC", 
            0, 
            &clocks_per_sec,
            sizeof(u32)
        ));
    }

    double ms_per_cycle = 1.0 / (clocks_per_sec / 1000);
    double time_ms = cc_count * ms_per_cycle;
    printf("## DPU cycles  = %g\n", cc_count);
    printf("## DPU cycles (ms) = %g [@%u MHz]\n", time_ms, clocks_per_sec / 1000000);
#endif
// #elif INSTRUCTIONS
// #ifdef CSV
//     printf("results_and_timings(instructions), %d, %d, %d, %.0f", nr_of_dpus, NR_TASKLETS, num_samples, cc_count);
// #else 

// #endif
// #endif
	
// #ifdef CSV
//     printf(", ");
//     print2(&timer, 2, p.n_reps);
//     printf(", ");
//     print2(&timer, 3, p.n_reps);
//     printf(", ");
//     print2(&timer, 4, p.n_reps);
//     printf(", ");
//     print2(&timer, 5, p.n_reps);
// #else 
//     printf("CPU-DPU transfers ");
//     print(&timer, 2, p.n_reps);
//     printf("\nDPU kernel ");
//     print(&timer, 3, p.n_reps);
//     printf("\nDPU-CPU transfers ");
//     print(&timer, 4, p.n_reps);
//     printf("\nCPU prediction ");
//     print(&timer, 5, p.n_reps);
// #endif 


    // Deallocation
    DPU_ASSERT(dpu_free(dpu_set)); // Deallocate DPUs
	
//     return 0;
}
