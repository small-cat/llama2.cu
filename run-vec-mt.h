#ifndef __RUN_VEC_MT_H__
#define __RUN_VEC_MT_H__

#include <stdio.h>

#include <pthread.h>
#include <stdatomic.h>

// #define MIN(a, b) ((a) < (b) ? (a) : (b))

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// ----------------------------------------------------------------------------
// for threads
typedef struct ComputeStateShared {
  int n_threads;

  // synchronization primitives
  atomic_int n_barrier;
  atomic_int n_barrier_passed;

  atomic_int current_chunk; // currently processing chunk during mul_mat, shared between all the threads
} ComputeStateShared;

typedef struct ComputeState {
  Transformer *transformer;
  int token;    // the token index
  int pos;

  pthread_t tid;
  int ith;
  struct ComputeStateShared *shared;
} ComputeState;

typedef struct ComputeParams {
  int ith, nth;
  ComputeStateShared *shared;
} ComputeParams;

// ----------------------------------------------------------------------------
// function definitions
void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);
void memory_map_weights(TransformerWeights* w, Config* p, float* ptr,
                        int shared_weights);
void read_checkpoint(char* checkpoint, Config* config,
                     TransformerWeights* weights, int* fd, float** data,
                     ssize_t* file_size);
void build_transformer(Transformer *t, char* checkpoint_path);
void free_transformer(Transformer* t);

// operator implementations
void rmsnorm(float* o, float* x, float* weight, int size, ComputeParams* params);
void softmax(float* x, int size);
void matmul(float* xout, float* x, float* w, int n, int d,
            ComputeParams* params);
void RoPe_rotation(int pos, RunState* s, int dim, int kv_dim, int head_size, ComputeParams* params);
void multi_head_attention(int pos, Config* p, RunState* s, int kv_dim,
                          int kv_mul, int head_size, int loff,
                          ComputeParams* params);
void accum(float* a, float* b, int size, ComputeParams* params);
void f_silu_elementwise_mul_w3(RunState* s, int hidden_dim,
                               ComputeParams* params);

float* forward(Transformer* transformer, int token, int pos, ComputeParams *params);
int compare_tokens(const void* a, const void* b);
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
char* decode(Tokenizer* t, int prev_token, int token);
void safe_printf(char *piece);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, int n_threads);
void read_stdin(const char* guide, char* buffer, size_t bufsize);
void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
          char* cli_user_prompt, char* cli_system_prompt, int steps,
          int n_threads);

// sampling
int sample_argmax(float* probabilities, int n);
int sample_mult(float* probabilities, int n, float coin);
int compare(const void* a, const void* b);
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin);
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);
unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);
int sample(Sampler* sampler, float* logits);

// create threads for computing llama
// @param token the index in vocabulary table
// @param n_threads number of threads
void compute_forward(Transformer* transformer, int token, int pos,
                     int n_threads);

#endif 