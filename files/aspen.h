#ifndef _ASPEN_H_
#define _ASPEN_H_
#define _GNU_SOURCE
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>

#define MAX_TENSOR_DIMS 8
#define MAX_STRING_LEN 256
#define MAX_PARENT_NINST_NUM (1<<16) // 65536
#define MAX_NUM_GPUS 16
#define NINST_H_MIN (64)
#define NINST_W_MIN (12)
#define MEM_ALIGN 64

typedef enum {NINST_NOT_READY, NINST_READY, NINST_COMPLETED, NUM_NINST_STATES} NINST_STATE;
typedef enum {NO_LAYER_TYPE, INPUT_LAYER, CONV_LAYER, FC_LAYER,
 RESIDUAL_LAYER, BATCHNORM_LAYER, YOLO_LAYER, APPEND_LAYER, ACTIVATION_LAYER, MAXPOOL_LAYER, AVGPOOL_LAYER,
 ROUTE_LAYER, SOFTMAX_LAYER,
 MATMUL_LAYER, LAYERNORM_LAYER, K_ATTENTION_LAYER, V_ATTENTION_LAYER,
 NUM_LAYER_ELEMENTS} LAYER_TYPE;
typedef enum {
    OUT_W, OUT_H, OUT_C, BATCH, IN_W, IN_H, IN_C, WEIGHT_W, WEIGHT_H, SUB_C, STRIDE, PADDING, DILATION, GROUPS,
    NUM_HIDDEN, NUM_HEAD, NUM_SEQ, MAT_M, MAT_N, MAT_K, SUB_M, MASKED,
    FORM_BYTES, NUM_PARAM_ELEMENTS
} LAYER_PARAMS;
typedef enum {NULL_TENSOR, OUTPUT_TENSOR, INPUT_TENSOR, WEIGHT_TENSOR, BIAS_TENSOR, COL_IDX_TENSOR, ANCHOR_TENSOR,
    BN_VAR_TENSOR, BN_MEAN_TENSOR, BN_WEIGHT_TENSOR, NUM_TENSORS} LAYER_TENSORS;
typedef enum {PARENT_NONE, PARENT_0, PARENT_1, PARENT_WEIGHT, NUM_PARENT_ELEMENTS} LAYER_PARENTS;
typedef enum {NO_ACTIVATION, SIGMOID, LINEAR, TANH, RELU, LEAKY_RELU, ELU, SELU, GELU, GELU_ACCURATE, NUM_ACTIVATIONS} LAYER_ACT;
typedef enum {RPOOL_DNN, RPOOL_LAYER_TYPE, RPOOL_LAYER_IDX, RPOOL_NASM, RPOOL_ASE, NUM_RPOOL_CONDS} RPOOL_CONDS;

extern char *ninst_state_str [NUM_NINST_STATES];
extern char *layer_type_str [NUM_LAYER_ELEMENTS];
extern char *param_type_str[NUM_PARAM_ELEMENTS];
extern char *tensor_type_str[NUM_TENSORS];
extern char *parent_type_str[NUM_PARENT_ELEMENTS];
extern char *activation_type_str [NUM_ACTIVATIONS];
extern char *rpool_cond_str [NUM_RPOOL_CONDS];

typedef struct aspen_dnn_t aspen_dnn_t;
typedef struct aspen_layer_t aspen_layer_t;
typedef struct aspen_tensor_t aspen_tensor_t;

typedef struct ninst_t ninst_t; // Ninst - ASPEN Graph Nodes
typedef struct nasm_t nasm_t;   // Nasm - ASPEN Graph
typedef struct nasm_ldata_t nasm_ldata_t; // Dynamic layer data

typedef struct rpool_t rpool_t; // Ready pool
typedef struct rpool_queue_t rpool_queue_t;
typedef struct rpool_queue_group_t rpool_queue_group_t;

typedef struct dse_t dse_t;     // Distributed scheduling engine
typedef struct dse_group_t dse_group_t;

aspen_dnn_t *apu_create_dnn(char *input_path, char *data_path);
void apu_destroy_dnn(aspen_dnn_t *dnn);
void apu_save_dnn_to_file(aspen_dnn_t *dnn, char *filename);
aspen_dnn_t *apu_load_dnn_from_file(char *filename);

nasm_t *apu_generate_nasm (aspen_dnn_t *dnn, unsigned int batch_size, unsigned int num_iter);
nasm_t *apu_generate_transformer_nasm (aspen_dnn_t *dnn, unsigned int batch_size, unsigned int seq_num, unsigned int num_iter);
nasm_t *apu_create_nasm(aspen_dnn_t *dnn, unsigned int min_ninst_per_ldata, unsigned int batch_size);
nasm_t *apu_create_transformer_nasm(aspen_dnn_t *dnn, unsigned int min_ninst_per_ldata, unsigned int batch_size, unsigned int seq_num);
void apu_destroy_nasm(nasm_t *nasm);
nasm_t *apu_load_nasm_from_file(char *filename, aspen_dnn_t *dnn);
void apu_save_nasm_to_file(nasm_t *nasm, char *filename);

rpool_t *rpool_init ();
void rpool_destroy (rpool_t *rpool);
void rpool_add_nasm_raw_input (rpool_t *rpool, nasm_t* nasm, void* input_data);
void rpool_add_nasm (rpool_t *rpool, nasm_t* nasm, char *input_filename);
void rpool_reset (rpool_t *rpool);
void rpool_reset_nasm (rpool_t *rpool, nasm_t *nasm);

dse_group_t *dse_group_init (unsigned int num_ase);
void dse_group_set_rpool (dse_group_t *dse_group, rpool_t *rpool);
void dse_group_destroy (dse_group_t *dse_group);
void dse_group_run (dse_group_t *dse_group);
void dse_group_stop (dse_group_t *dse_group);
void dse_group_run_until_nasm_completion (dse_group_t *dse_group, nasm_t *nasm);
void dse_wait_for_nasm_completion (nasm_t *nasm);
unsigned int dse_check_nasm_completion (nasm_t *nasm);
void *dse_get_ldata_result (nasm_t *nasm, unsigned int ldata_idx, LAYER_PARAMS *order);
void *dse_get_nasm_result (nasm_t *nasm, LAYER_PARAMS *order);

void print_aspen_build_info(void);
void print_dnn_info (aspen_dnn_t *dnn, int print_data);
void print_layer_info (aspen_layer_t *layer, int print_data);
void print_tensor_info (aspen_tensor_t *tensor, int print_data);
void print_nasm_info (nasm_t *nasm, int print_ninst, int print_data);
void print_ldata_info (nasm_ldata_t *ldata, int print_ninst, int print_data);
void print_ninst_info (ninst_t *ninst, int print_data);
void print_rpool_queue_info (rpool_queue_t *rpool_queue);
void print_rpool_queue_group_info (rpool_queue_group_t *rpool_queue_group);
void print_rpool_info (rpool_t *rpool);

#endif /* _ASPEN_H_ */