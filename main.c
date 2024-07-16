#include <stdio.h>

#include "common.h"
#include "run-vec-mt.h"

static GptParams *g_params;
int main(int argc, char *argv[]) {
  GptParams params = {
    0,
    0,
    256,
    1.0f,
    0.9f,
    NULL,
    DEFAULT_TOKENIZER_PATH,
    NULL,
    "generate",
    NULL
  };
  g_params = &params;

  if (!gpt_params_parse(argc, argv, &params)) {
    exit(EXIT_FAILURE);
  }

  // parameter validation/overrides
  if (params.seed <= 0) params.seed = (unsigned int)time(NULL);
  if (params.temperature < 0.0) params.temperature = 0.0;
  if (params.topp < 0.0 || 1.0 < params.topp) params.topp = 0.9;
  if (params.n_predict < 0) params.n_predict = 0;
  if (params.n_threads == 0) params.n_threads = cpu_get_num_math();

  show_settings(params);

  // build the Transformer via the model .bin file
  Transformer transformer;
  int32_t steps = params.n_predict;
  build_transformer(&transformer, params.checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len)
    steps = transformer.config.seq_len;  // ovrerride to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, params.tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, params.temperature,
                params.topp, params.seed);

  // run!
  if (strcmp(params.mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, params.prompt, steps, params.n_threads);
  } else if (strcmp(params.mode, "chat") == 0) {
    chat(&transformer, &tokenizer, &sampler, params.prompt, params.system_prompt, steps, params.n_threads);
  } else {
    fprintf(stderr, "unknown mode: %s\n", params.mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}