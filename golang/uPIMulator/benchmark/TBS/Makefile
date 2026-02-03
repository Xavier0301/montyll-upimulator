DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin
TBTC_DIR := tbtc-htm
NR_DPUS ?= 1# max nr of dpus. The model parallelism will scale accordingly
NR_TASKLETS ?= 11# => 11 is the minimum recommended number of tasklets. Keeping it as low as possible helps with model sharing
PRINT ?= 0# Can be 1
PERF ?= NO# Can be INSTRUCTIONS or CYCLES
FORMAT ?= NO# Can be CSV
DPU_CHECK_CONNS ?= 0


define conf_filename
	${BUILDDIR}/.DPUS_$(0)_PRINT_$(1)_FORMAT_$(2)_PERF_$(3)_DPU_CHECK_CONNS_$(4).conf
endef
CONF := $(call conf_filename,${NR_DPUS},${PRINT},${FORMAT},${PERF},${DPU_CHECK_CONNS})

HOST_TARGET := ${BUILDDIR}/host_code
DPU_TARGET := ${BUILDDIR}/dpu_code

COMMON_INCLUDES := support

EXCLUDED_SRCS := ${TBTC_DIR}/main.c \
                 ${TBTC_DIR}/scale_out.c \
                 ${TBTC_DIR}/omp_test.c

ALL_HOST_SOURCES := $(wildcard ${TBTC_DIR}/*.c ${HOST_DIR}/*.c) # collect all sources..
HOST_SOURCES := $(filter-out $(EXCLUDED_SRCS), $(ALL_HOST_SOURCES))
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c) ${TBTC_DIR}/lm_parameters.c


.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}
HOST_FLAGS := ${COMMON_FLAGS} -std=c11 -O3 -lm `dpu-pkg-config --cflags --libs dpu` \
	-DNR_DPUS=${NR_DPUS} -DNR_TASKLETS=${NR_TASKLETS} -DPRINT=${PRINT} -D${PERF} -D${FORMAT} -DDPU_CHECK_CONNS=${DPU_CHECK_CONNS}
DPU_FLAGS := ${COMMON_FLAGS} -Wframe-larger-than=256 -O2 -DNR_TASKLETS=${NR_TASKLETS} -DPRINT=${PRINT} -D${PERF} -D${FORMAT} -DDPU_CHECK_CONNS=${DPU_CHECK_CONNS}

all: ${HOST_TARGET} ${DPU_TARGET}

${CONF}:
	$(RM) $(call conf_filename,*,*)
	touch ${CONF}

${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES} ${CONF}
	$(CC) -o $@ ${HOST_SOURCES} ${HOST_FLAGS}

${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

assembler: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -S -fverbose-asm ${DPU_SOURCES}

clean:
	$(RM) -r $(BUILDDIR)

test: all
	./${HOST_TARGET}

# rsync -a ./PIMbthowen2/ upmemcloud5:/home/upmem0013/xservot/PIMbthowen2
