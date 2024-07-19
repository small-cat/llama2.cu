/* Inference for Llama-2 Transformer model in pure C */

#include <ctype.h>
#include <math.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#include <assert.h>
#include <sched.h>

#include "common.h"
#include "mt-vec.h"
#include "run-vec-mt.h"

// @brief set a barrier to synchronize threads
static void thread_barrier(ComputeStateShared *shared) {
  if (shared->n_threads == 1) return;

#ifdef USE_OPENMP
  #pragma omp barrier
#else
  atomic_int* n_barrier = &shared->n_barrier;
  atomic_int* n_barrier_passed = &shared->n_barrier_passed;

  int n_threads = shared->n_threads;
  int passed_old = atomic_load(n_barrier_passed);

  if (atomic_fetch_add(n_barrier, 1) == n_threads - 1) {
    // thread increase n_barrier one by one, if n_barrier == n_threads - 1
    // the last thread arrived
    atomic_store(n_barrier, 0);
    atomic_fetch_add(n_barrier_passed, 1);
  } else {
    // wait for other threads until all the threads reached the barrier
    const int n_spin_before_sleep = 100000;
    while (1) {
      for (int i = 0; i < n_spin_before_sleep; i++) {
        if (atomic_load(n_barrier_passed) != passed_old) {
          // the last thread had arrived and changed n_barrier_passed
          return;
        }

      #if defined(__SSE3__)
          _mm_pause();
      #endif
      }

      // relinguish the cpu and let other thread to run, lower the current priority
      sched_yield();  // a system call defined in linux
    }
  }
#endif
}

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void show_config(Config* conf) {
  printf("-----------------------CONFIGS---------------------\n");
  printf("  dim: %d\n", conf->dim);
  printf("  hidden_dim: %d\n", conf->hidden_dim);
  printf("  n_layers: %d\n", conf->n_layers);
  printf("  n_heads: %d\n", conf->n_heads);
  printf("  n_kv_heads: %d\n", conf->n_kv_heads);
  printf("  vocab_size: %d\n", conf->vocab_size);
  printf("  seq_len: %d\n", conf->seq_len);
  printf("---------------------------------------------------\n\n");
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    show_config(&t->config);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

static void print_data(float* ptr, int start, int end, char* matrix_name,
                       ComputeParams* params) {
  if (params->ith != 0) return;
  printf("\n------------------%s---------------\n", matrix_name);
  for (int i = start; i < end; i++) {
    printf("%f ", ptr[i]);
  }
  printf("\n");
}

#ifdef USE_VECTORIZE
inline static void llama_vec_scale_mul_f32(const int n, float* x, float* w,
                                           const float scale) {
  // for (int i = 0; i < n; i++) {
  //   x[i] = x[i] * scale * w[i];
  // }
  const int np = (n & ~(GGML_F32_STEP - 1));  // n / GGML_F32_STEP
  GGML_F32_VEC vx = GGML_F32_VEC_SET1(
      scale);  // vx is a vector contains number of 16 floats all equal to scale
  GGML_F32_VEC aw[GGML_F32_ARR];  // store input weights
  GGML_F32_VEC ax[GGML_F32_ARR];  // store input x

  // GGML_F32_STEP = GGML_F32_ARR * GGML_F32_EPR
  for (int i = 0; i < np; i += GGML_F32_STEP) {
    for (int j = 0; j < GGML_F32_ARR; j++) {
      ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
      ax[j] = GGML_F32_VEC_MUL(ax[j], vx);  // xi * scale

      aw[j] = GGML_F32_VEC_LOAD(w + i + j * GGML_F32_EPR);
      ax[j] = GGML_F32_VEC_MUL(ax[j], aw[j]); // xi * scale * wi

      GGML_F32_VEC_STORE(x + i + j * GGML_F32_EPR, ax[j]);
    }
  }

  // the remains elements
  for (int i = np; i < n; i++) {
    x[i] = x[i] * scale * w[i];
  }
}

// 1 / sqrt(sum(xi ^ xi)/size + epsilon) * xi * wi
// llama.cpp wrap above to llm_build_norm, consist of three node
// 1. rms_norm
// 2. mul weight
// 3. add bias if bias is not null
void rmsnorm(float* o, float* x, float* weight, int size, ComputeParams* params) {
  // rmsnorm runs as single thread
  if (params->ith != 0) return;

  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
    o[j] = x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

  llama_vec_scale_mul_f32(size, o, weight, ss);
}
#else
void rmsnorm(float* o, float* x, float* weight, int size, ComputeParams* params) {
  if (params->ith != 0) return;
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
#endif

void softmax(float* x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

#ifdef USE_VECTORIZE
static void vec_dot_f32(int n, float* xout, float* x, float* y) {
  float sumf = 0.0f;
  int np = (n & ~(GGML_F32_STEP - 1));    // n / GGML_F32_STEP
  GGML_F32_VEC sum[GGML_F32_ARR] = {GGML_F32_VEC_ZERO};

  GGML_F32_VEC ax[GGML_F32_ARR];
  GGML_F32_VEC ay[GGML_F32_ARR];
  for (int i = 0; i < np; i += GGML_F32_STEP) {
    for (int j = 0; j < GGML_F32_ARR; j++) {
      ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);

      sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0...sumN to sum0
  GGML_F32_VEC_REDUCE(sumf, sum);

  // remainders
  for (int i = np; i < n; ++i) {
    sumf += x[i] * y[i];
  }

  *xout = sumf;
}

static void compute_mul_mat_one_chunk(float* xout, float* x, float* w, int n,
                                      int d, int ir0_start, int ir0_end,
                                      int ir1_start, int ir1_end) {
  // w(d, n) dot x(n,) -> xout(d, )

  // block tiling, for gemv, 16 * 1
  // int blck_0 = 1;
  // int blck_1 = 16;
  // int col_stride = row_size;

  for (int iir1 = ir1_start; iir1 < ir1_end; iir1 += 1) { // per line
    float* w_ptr = w + iir1 * n;
    float* x_ptr = x;
    float* dst_col = xout + iir1;

    vec_dot_f32(n, dst_col, w_ptr, x_ptr);
  }
}

void matmul(float* xout, float* x, float* w, int n, int d, ComputeParams* params) {
  int ith = params->ith;
  int nth = params->nth;

  if (ith == 0) {
    // the first unprocessed chunk is nth
    atomic_store(&params->shared->current_chunk, nth);
  }
  thread_barrier(params->shared);
  // all threads could fetch current_chunk and know the value

  // gemv, (d, n) dot (n, ) -> (d, )
  // split dest matrix along with d to chunks, so each thread could process one
  // chunk, which means, each thread could process several lines gemv

  // the following algorithm from llama.cpp
  int chunk_size = 64;  // for gemv, a resonable chunk size ?
  // int num_rows_per_vec_dot = 1;

  int n_chunks = (d + chunk_size - 1) / chunk_size;   // number of chunks along xout rows (d,)
  if (n_chunks < nth * 4) {
    // https://github.com/ggerganov/llama.cpp/pull/6915
    // seems more fast by this way
    n_chunks = nth;   // parallelize by x(d, n) rows
  }

  // calc the number of elements in each chunk
  int dr = (d + n_chunks - 1) / n_chunks;

  // if set 4 threads, and n_chunks is 8
  // ideally, 
  // thread 0 process 0th and 4th chunk
  // thread 1 process 1th and 5th chunk
  // thread 2 process 2th and 6th chunk
  // thread 3 process 3th and 7th chunk
  // if thread 1 process slowly and thread 2 process fast, thread 2 could process
  // 2th, 5th, 6th chunk, and thread 1 process only 1th chunk
  int current_chunk = ith;
  while (current_chunk < n_chunks) {
    int ith_col = current_chunk % n_chunks; // current thread should process which chunk along dest rows

    // dest matrix has only one column (gemv)
    int ir0_start = 0;
    int ir0_end = 1;

    int ir1_start = ith_col * dr;
    int ir1_end = MIN(ir1_start + dr, d);

    compute_mul_mat_one_chunk(xout, x, w, n, d, ir0_start, ir0_end, ir1_start, ir1_end);

    if (nth >= n_chunks)
      break;

    current_chunk = atomic_fetch_add(&params->shared->current_chunk, 1);
  }
}
#else
void matmul(float* xout, float* x, float* w, int n, int d,
            ComputeParams* params) {
  if (params->ith != 0) return;
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
// #pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}
#endif

#ifdef USE_VECTORIZE
void RoPe_rotation(int pos, RunState* s, int dim, int kv_dim, int head_size,
                   ComputeParams* params) {
  int ith = params->ith;
  int nth = params->nth;

  if (ith == 0) {
    atomic_store(&params->shared->current_chunk, nth);
  }
  thread_barrier(params->shared);

  int chunk_size = 64;  // each thread process 32*2 elements, v0, v1
  int n_chunk = (dim + chunk_size - 1) / chunk_size;

  if (nth > n_chunk) {
    fprintf(stderr, "threads number %d greater than %d chunks\n", nth, n_chunk);
    exit(EXIT_FAILURE);
  }

  int current_chunk = ith;
  while (current_chunk < n_chunk) {
    int cur_idx = current_chunk % n_chunk;
    int ir0 = cur_idx * chunk_size;
    int ir1 = MIN(ir0 + chunk_size, dim);

    for (int i = ir0; i < ir1; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1;
      for (int v = 0; v < rotn; v++) {
        float* vec = v == 0 ? s->q : s->k;
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    current_chunk = atomic_fetch_add(&params->shared->current_chunk, 1);
  }
}
#else
void RoPe_rotation(int pos, RunState* s, int dim, int kv_dim, int head_size,
                   ComputeParams* params) {  // s->q, s->k, freq_cis_real_row,
                                             // freq_cis_imag_row,
                                             // p->n_heads, head_size) {
  if (params->ith != 0) return;

  for (int i = 0; i < dim; i += 2) {
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
      float* vec = v == 0 ? s->q : s->k;  // the vector to rotate (query or key)
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}
#endif

void multi_head_attention(int pos, Config* p, RunState* s, int kv_dim,
                          int kv_mul, int head_size, int loff, ComputeParams* params) {
  if (params->ith != 0) return;

  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < p->n_heads; h++) {
    // get the query vector for this head
    float* q = s->q + h * head_size;
    // attention scores for this head
    float* att = s->att + h * p->seq_len;
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
      // get the key vector for this head and at this timestep
      float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      // calculate the attention score as the dot product of q and k
      float score = 0.0f;
      for (int i = 0; i < head_size; i++) {
        score += q[i] * k[i];
      }
      score /= sqrtf(head_size);
      // save the score to the attention buffer
      att[t] = score;
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    softmax(att, pos + 1);

    // weighted sum of the values, store back into xb
    float* xb = s->xb + h * head_size;
    memset(xb, 0, head_size * sizeof(float));
    for (int t = 0; t <= pos; t++) {
      // get the value vector for this head and at this timestep
      float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      // get the attention weight for this timestep
      float a = att[t];
      // accumulate the weighted value into xb
      for (int i = 0; i < head_size; i++) {
        xb[i] += a * v[i];
      }
    }
  }
}

void accum(float* a, float* b, int size, ComputeParams* params) {
  if (params->ith != 0) return;

  for (int i = 0; i < size; i++) {
    a[i] += b[i];
  }
}

void f_silu_elementwise_mul_w3(RunState* s, int hidden_dim, ComputeParams* params) {
  if (params->ith != 0) {
    return;
  }

  for (int i = 0; i < hidden_dim; i++) {
    float val = s->hb[i];
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + expf(-val)));
    // elementwise multiply with w3(x)
    val *= s->hb2[i];
    s->hb[i] = val;
  }
}

float* forward(Transformer* transformer, int token, int pos, ComputeParams *params) {
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim, params);
        thread_barrier(params->shared);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim, params);
        thread_barrier(params->shared);

        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim, params);
        thread_barrier(params->shared);

        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim, params);
        thread_barrier(params->shared);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        RoPe_rotation(pos, s, dim, kv_dim, head_size, params);
        thread_barrier(params->shared);
        // print_data(s->q, 100, 120, "RoPe_rotation-q", params);

        // multihead attention. iterate over all heads
        multi_head_attention(pos, p, s, kv_dim, kv_mul, head_size, loff, params);

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim, params);
        thread_barrier(params->shared);

        // ffn_inp = ggml_add(cur, inpSA)
        // residual connection back into x
        accum(x, s->xb2, dim, params);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim, params);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim, params);
        thread_barrier(params->shared);

        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim, params);
        thread_barrier(params->shared);

        // SwiGLU non-linearity
        f_silu_elementwise_mul_w3(s, hidden_dim, params);

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim, params);
        thread_barrier(params->shared);

        // in llama.cpp: llm_build_ffn
        // w1 -> gate, w3 -> up, w2 -> down
        // (silu(in * gate) * (in * up)) * down
        // (silu(in * w1) * (in * w3)) * w2

        // residual connection
        accum(x, s->xb, dim, params);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim, params);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size, params);
    thread_barrier(params->shared);

    return s->logits;
}

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

static void* llama_compute_thread(void *data) {
    ComputeState* state = (ComputeState *) data;
    Transformer* transformer = state->transformer;
    int token = state->token;
    int pos = state->pos;

    set_numa_thread_affinity(state->ith);

    ComputeParams params = {
        .ith = state->ith,
        .nth = state->shared->n_threads,
        .shared = state->shared,
    };

    // if (state->ith != 0) {
    //     // TODO: multi-threads does not implement, only use the first thread
    //     return 0;
    // }

    forward(transformer, token, pos, &params);

    return 0;
}

void compute_forward(Transformer *transformer, int token, int pos, int n_threads) {
    assert(n_threads >= 1);

    ComputeStateShared state_shared = {
        n_threads,
        0,          // n_barrier
        0,          // n_barrier_passed
        0           // current_chunk
    };

    // apply a buffer from stack
    ComputeState* workers = (ComputeState *)alloca(sizeof(ComputeState) * n_threads); 
    for (int i = 0; i < n_threads; ++i) {
        workers[i] = (ComputeState) {
            .transformer = transformer,
            .token = token,
            .pos = pos,
            .tid = 0,
            .ith = i,
            .shared = &state_shared,
        };
    }

    for (int i = 1; i < n_threads; ++i) {
        const int rc = pthread_create(&workers[i].tid, NULL, llama_compute_thread, &workers[i]);
        assert(rc == 0);
    }

    // main thread is also a worker thread
    llama_compute_thread(&workers[0]);

    if (n_threads > 1) {
        for (int i = 1; i < n_threads; ++i) {
            const int rc = pthread_join(workers[i].tid, NULL);
            assert(rc == 0);
        }
    }

    clear_numa_thread_affinity();
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, int n_threads) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // TODO(@wuzhenyu): how to statistic prefill time and decode time when use multi-threads
    // TODO(@wuzhenyu): separate prefill and decode, prefill use prompt_tokens, but decode phase does not 

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        // token means the idx in the vocabulary table
        // float* logits = forward(transformer, token, pos);
        compute_forward(transformer, token, pos, n_threads);
        float* logits = transformer->state.logits;

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
          char* cli_user_prompt, char* cli_system_prompt, int steps,
          int n_threads) {
  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are soomewhat haphazardly and unsafely set atm
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[1152];
  int num_prompt_tokens = 0;
  int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
  int user_idx;

  // start the main loop
  int8_t user_turn = 1;  // user starts
  int next;              // will store the next token in the sequence
  int token;  // stores the current token to feed into the transformer
  int prev_token;
  int pos = 0;  // position in the sequence
  while (pos < steps) {
    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt,
                     sizeof(system_prompt));
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
      }
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User: ", user_prompt, sizeof(user_prompt));
      }
      // render user/system prompts into the Llama 2 Chat schema
      if (pos == 0 && system_prompt[0] != '\0') {
        char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
        sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
      } else {
        char user_template[] = "[INST] %s [/INST]";
        sprintf(rendered_prompt, user_template, user_prompt);
      }
      // encode the rendered prompt into tokens
      encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens,
             &num_prompt_tokens);
      user_idx = 0;  // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS (=2) token ends the Assistant turn
    if (token == 2) {
      user_turn = 1;
    }

    // forward the transformer to get logits for the next token
    // float* logits = forward(transformer, token, pos);
    compute_forward(transformer, token, pos, n_threads);
    float* logits = transformer->state.logits;
    next = sample(sampler, logits);
    pos++;

    if (user_idx >= num_prompt_tokens && next != 2) {
      // the Assistant is responding, so print its output
      char* piece = decode(tokenizer, token, next);
      safe_printf(
          piece);  // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
    }
    if (next == 2) {
      printf("\n");
    }
  }
  printf("\n");
  free(prompt_tokens);
}
