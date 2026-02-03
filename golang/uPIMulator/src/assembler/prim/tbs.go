package prim

/*
#cgo CFLAGS: -I${SRCDIR}/../../../../../../pim-tbtc-htm/tbtc-htm -I${SRCDIR}/../../../../../../pim-tbtc-htm/support
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"

#include "types.h"

#include "tensor.h"
#include "grid_environment.h"
#include "sensor_module.h"
#include "learning_module.h"
#include "motor_policy.h"

// Cgo cant see the macro for some reason
static uint8_t repr_u8_helper(float val) {
    return REPR_u8(val);
}

// This shouldnt be necessary but coudl be in the future for aligned DMA accesses
static inline uint32_t round_up_8(uint32_t n) {
    return (n + 7) & ~7;
}

static void serialize_model_params(dpu_model_params_t* p, void* dest) {
    memcpy(dest, p, sizeof(dpu_model_params_t));
}

static void serialize_mram_content(mram_content_t* p, void* dest) {
    memcpy(dest, p, sizeof(mram_content_t));
}
*/
import "C"

import (
	"errors"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
	"unsafe"
)

const (
	NUM_EXTERNAL_LMS = 20
	OUT_CELL_LOG_DIM = 10
)

type Tbs struct {
	num_dpus       int
	num_tasklets   int
	num_executions int

	lm                  C.learning_module
	sm                  C.grid_sm
	movement            C.vec2d
	features            C.features_t
	external_o_activity C.lmat_u32

	model_params C.dpu_model_params_t
	mram_size    C.mram_content_t
	mram_addr    C.mram_content_t
}

func (this *Tbs) Init(command_line_parser *misc.CommandLineParser) {
	this.num_dpus = int(command_line_parser.IntParameter("num_channels")) *
		int(command_line_parser.IntParameter("num_ranks_per_channel")) *
		int(command_line_parser.IntParameter("num_dpus_per_rank"))
	this.num_tasklets = int(command_line_parser.IntParameter("num_tasklets"))
	this.num_executions = 1

	num_cols := C.u32(1024)
	C.init_sensor_module(&this.sm, C.GRID_ENV_MIN_VALUE, C.GRID_ENV_MAX_VALUE, num_cols)
	C.init_features(&this.features, this.sm.pooler.params.num_minicols, C.u32(this.sm.pooler.params.top_k))

	C.lmat_u32_init(&this.external_o_activity, C.u8(NUM_EXTERNAL_LMS), C.u8(OUT_CELL_LOG_DIM-5))

	htm_params := C.htm_params_t{
		permanence_threshold:      C.repr_u8_helper(0.5),
		segment_spiking_threshold: 15,
		perm_increment:            C.repr_u8_helper(0.06),
		perm_decrement:            C.repr_u8_helper(0.04),
		perm_decay:                1,
	}

	ext_htm_params := C.extended_htm_params_t{
		feedforward_permanence_threshold: C.repr_u8_helper(0.5),
		context_permanence_threshold:     C.repr_u8_helper(0.5),
		feedforward_activation_threshold: 3,
		context_activation_threshold:     18,
		min_active_cells:                 10,
	}

	output_p := C.output_layer_params_t{
		cells:                     1 << OUT_CELL_LOG_DIM,
		log_cells:                 OUT_CELL_LOG_DIM,
		internal_context_segments: 6,
		external_context_segments: 6,
		external_cells:            1 << OUT_CELL_LOG_DIM,
		log_external_cells:        OUT_CELL_LOG_DIM,
		external_lms:              NUM_EXTERNAL_LMS,
		htm:                       htm_params,
		extended_htm:              ext_htm_params,
	}

	features_p := C.feature_layer_params_t{
		cols:              C.u16(num_cols),
		cells:             8,
		feature_segments:  6,
		location_segments: 6,
		htm:               htm_params,
	}

	location_p := C.location_layer_params_t{
		cols:              C.u32(num_cols),
		log_cols_sqrt:     5, // log2(sqrt(1024))
		cells:             8,
		location_segments: 6,
		feature_segments:  6,
		log_scale:         C.uvec2d{x: 0, y: 0},
		htm:               htm_params,
	}

	C.init_learning_module(&this.lm, output_p, features_p, location_p)

	this.model_params = C.dpu_model_params_t{
		output:   output_p,
		features: features_p,
		location: location_p,
	}

	this.mram_size = C.mram_content_t{
		f_feature_context:      C.feature_layer_get_feature_context_footprint_bytes(features_p),
		f_location_context:     C.feature_layer_get_location_context_footprint_bytes(features_p),
		l_location_context:     C.location_layer_get_location_context_footprint_bytes(location_p),
		l_feature_context:      C.location_layer_get_feature_context_footprint_bytes(location_p),
		o_internal_context:     C.output_layer_get_internal_context_footprint_bytes(output_p),
		o_external_context:     C.output_layer_get_external_context_footprint_bytes(output_p),
		o_feedforward:          C.output_layer_get_feedforward_footprint_bytes(output_p),
		input_movement:         C.u32(unsafe.Sizeof(this.movement)),
		input_features:         C.round_up_8(num_cols * C.u32(unsafe.Sizeof(C.u32(0)))),
		external_o_activity:    C.lmat_u32_count(&this.external_o_activity) * C.u32(unsafe.Sizeof(C.u32(0))),
		output:                 C.u32(output_p.cells>>5) * C.u32(unsafe.Sizeof(C.u32(0))),
		f_feature_spike_cache:  C.feature_layer_get_feature_segments_spike_count_cache_bytes(features_p),
		f_location_spike_cache: C.feature_layer_get_location_segments_spike_count_cache_bytes(features_p),
		l_location_spike_cache: C.location_layer_get_location_segments_spike_count_cache_bytes(location_p),
		l_feature_spike_cache:  C.location_layer_get_feature_segments_spike_count_cache_bytes(location_p),
		o_internal_spike_cache: C.output_layer_get_internal_context_segments_spike_count_cache_bytes(output_p),
		o_external_spike_cache: C.output_layer_get_external_context_segments_spike_count_cache_bytes(output_p),
	}

	this.mram_addr = C.mram_content_t{f_feature_context: 0}

	accumulate := func(prevAddr C.u32, prevSize C.u32) C.u32 { return prevAddr + prevSize }

	this.mram_addr.f_location_context = accumulate(this.mram_addr.f_feature_context, this.mram_size.f_feature_context)
	this.mram_addr.l_location_context = accumulate(this.mram_addr.f_location_context, this.mram_size.f_location_context)
	this.mram_addr.l_feature_context = accumulate(this.mram_addr.l_location_context, this.mram_size.l_location_context)
	this.mram_addr.o_internal_context = accumulate(this.mram_addr.l_feature_context, this.mram_size.l_feature_context)
	this.mram_addr.o_external_context = accumulate(this.mram_addr.o_internal_context, this.mram_size.o_internal_context)
	this.mram_addr.o_feedforward = accumulate(this.mram_addr.o_external_context, this.mram_size.o_external_context)
	this.mram_addr.input_movement = accumulate(this.mram_addr.o_feedforward, this.mram_size.o_feedforward)
	this.mram_addr.input_features = accumulate(this.mram_addr.input_movement, this.mram_size.input_movement)
	this.mram_addr.external_o_activity = accumulate(this.mram_addr.input_features, this.mram_size.input_features)
	this.mram_addr.output = accumulate(this.mram_addr.external_o_activity, this.mram_size.external_o_activity)
	this.mram_addr.f_feature_spike_cache = accumulate(this.mram_addr.output, this.mram_size.output)
	this.mram_addr.f_location_spike_cache = accumulate(this.mram_addr.f_feature_spike_cache, this.mram_size.f_feature_spike_cache)
	this.mram_addr.l_location_spike_cache = accumulate(this.mram_addr.f_location_spike_cache, this.mram_size.f_location_spike_cache)
	this.mram_addr.l_feature_spike_cache = accumulate(this.mram_addr.l_location_spike_cache, this.mram_size.l_location_spike_cache)
	this.mram_addr.o_internal_spike_cache = accumulate(this.mram_addr.l_feature_spike_cache, this.mram_size.l_feature_spike_cache)
	this.mram_addr.o_external_spike_cache = accumulate(this.mram_addr.o_internal_spike_cache, this.mram_size.o_internal_spike_cache)

	this.movement = C.vec2d{x: 0, y: 0}
}

func (this *Tbs) toByteStream(ptr unsafe.Pointer, sizeBytes uintptr) *encoding.ByteStream {
	// C.GoBytes creates a copy of the C memory into a Go slice
	rawBytes := C.GoBytes(ptr, C.int(sizeBytes))

	stream := new(encoding.ByteStream)
	stream.Init()

	for _, b := range rawBytes {
		w := new(word.Word)
		w.Init(8)
		w.SetValue(int64(b))
		stream.Merge(w.ToByteStream())
	}
	return stream
}

func (this *Tbs) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	return map[string]*encoding.ByteStream{
		"p":          this.toByteStream(unsafe.Pointer(&this.model_params), unsafe.Sizeof(this.model_params)),
		"size_bytes": this.toByteStream(unsafe.Pointer(&this.mram_size), unsafe.Sizeof(this.mram_size)),
		"addresses":  this.toByteStream(unsafe.Pointer(&this.mram_addr), unsafe.Sizeof(this.mram_addr)),
	}
}

func (this *Tbs) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	if execution >= this.num_executions {
		err := errors.New("execution >= num executions")
		panic(err)
	} else if dpu_id >= this.num_dpus {
		err := errors.New("DPU ID >= num DPUs")
		panic(err)
	}

	return make(map[string]*encoding.ByteStream, 0)
}

func (this *Tbs) InputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	byteStream := new(encoding.ByteStream)
	byteStream.Init()

	// Helper to extract C memory pointers and Merge them byte-by-byte
	appendPart := func(ptr unsafe.Pointer, size C.u32) {
		if ptr != nil && size > 0 {
			slice := C.GoBytes(ptr, C.int(size))
			for _, b := range slice {
				this.appendByte(byteStream, b)
			}
		}
	}

	// CONCATENATE MRAM DATA: Order must match mram_addr calculation exactly
	appendPart(unsafe.Pointer(this.lm.feature_net.in_segments.feature_context), this.mram_size.f_feature_context)
	appendPart(unsafe.Pointer(this.lm.feature_net.in_segments.location_context), this.mram_size.f_location_context)
	appendPart(unsafe.Pointer(this.lm.location_net.in_segments.location_context), this.mram_size.l_location_context)
	appendPart(unsafe.Pointer(this.lm.location_net.in_segments.feature_context), this.mram_size.l_feature_context)
	appendPart(unsafe.Pointer(this.lm.output_net.in_segments.internal_context), this.mram_size.o_internal_context)
	appendPart(unsafe.Pointer(this.lm.output_net.in_segments.external_context), this.mram_size.o_external_context)
	appendPart(unsafe.Pointer(this.lm.output_net.in_segments.feedforward), this.mram_size.o_feedforward)
	appendPart(unsafe.Pointer(&this.movement), this.mram_size.input_movement)
	appendPart(unsafe.Pointer(this.features.active_columns), this.mram_size.input_features)
	appendPart(unsafe.Pointer(this.external_o_activity.data), this.mram_size.external_o_activity)

	// Output area (initialized to 0)
	for i := 0; i < int(this.mram_size.output); i++ {
		this.appendByte(byteStream, 0)
	}

	appendPart(unsafe.Pointer(this.lm.feature_net.spike_count_cache.feature_segments), this.mram_size.f_feature_spike_cache)
	appendPart(unsafe.Pointer(this.lm.feature_net.spike_count_cache.location_segments), this.mram_size.f_location_spike_cache)
	appendPart(unsafe.Pointer(this.lm.location_net.spike_count_cache.location_segments), this.mram_size.l_location_spike_cache)
	appendPart(unsafe.Pointer(this.lm.location_net.spike_count_cache.feature_segments), this.mram_size.l_feature_spike_cache)
	appendPart(unsafe.Pointer(this.lm.output_net.spike_count_cache.internal_context_segments), this.mram_size.o_internal_spike_cache)
	appendPart(unsafe.Pointer(this.lm.output_net.spike_count_cache.external_context_segments), this.mram_size.o_external_spike_cache)

	return 0, byteStream
}

func (this *Tbs) OutputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	// The simulator will read back from the DPU_MRAM_HEAP_POINTER + output_offset
	offset := int64(this.mram_addr.output)
	size := int(this.mram_size.output)

	byteStream := new(encoding.ByteStream)
	byteStream.Init()
	for i := 0; i < size; i++ {
		this.appendByte(byteStream, 0) // Allocate space for output
	}
	return offset, byteStream
}

func (this *Tbs) appendByte(stream *encoding.ByteStream, b byte) {
	w := new(word.Word)
	w.Init(8)
	w.SetValue(int64(b))
	stream.Merge(w.ToByteStream())
}

func (this *Tbs) NumExecutions() int {
	return this.num_executions
}
