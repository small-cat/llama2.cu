#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#include <sys/types.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF
#define DEFAULT_TOKENIZER_PATH "tokenizer.bin"

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// ----------------------------------------------------------------------------
// parameters
// it is c++11 specification initialize struct
typedef struct {
  unsigned long long seed ; // = 0;  // LLAMA_DEFAULT_SEED
  int32_t n_threads       ; // = 1; // cpu_get_num_math()
  int32_t n_predict       ; // = 256;
  float   temperature     ; // = 1.0f;
  float   topp            ; // = 0.9f;
  char*   checkpoint_path ; // = nullptr;
  char*   tokenizer_path  ; // = DEFAULT_TOKENIZER_PATH;
  char*   prompt          ; // = nullptr;
  char*   mode            ; // = "generate";
  char*   system_prompt   ; // = nullptr;
} GptParams;

void error_usage();
int gpt_params_parse(int argc, char ** argv, GptParams * params);

// utility
long time_in_ms();
int32_t cpu_get_num_math();
int32_t cpu_get_num_physical_cores();
void set_numa_thread_affinity(int thread_idx);
void clear_numa_thread_affinity();

void show_settings(GptParams params);

#endif 
