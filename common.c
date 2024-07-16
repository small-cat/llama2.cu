#include <stdio.h>
#include <unistd.h>

#include "common.h"

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --temperature <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  --topp <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s | --seed <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n | --n_predict <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z | --tokenizer <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  --threads <int>  number of threads to run\n");
}

int gpt_params_parse(int argc, char ** argv, GptParams * params) {
    if (argc < 2) {
        error_usage();
        return 0;
    }

    params->checkpoint_path = argv[1];

    char* arg = NULL;
    for (int i = 2; i < argc; i++) {
        arg = argv[i];

        if (strcmp(arg, "-s") == 0 || strcmp(arg, "--seed") == 0) {
            params->seed = atof(argv[++i]);
        } else if (strcmp(arg, "--temperature") == 0) {
            params->temperature = atof(argv[++i]);
        } else if (strcmp(arg, "--topp") == 0) {
            params->topp = atof(argv[++i]);
        } else if (strcmp(arg, "--n_predict") == 0 || strcmp(arg, "-n") == 0) {
            params->n_predict = atoi(argv[++i]);
        } else if (strcmp(arg, "-i") == 0) {
            params->prompt = argv[++i];
        } else if (strcmp(arg, "-z") == 0 || strcmp(arg, "--tokenizer") == 0) {
            params->tokenizer_path = argv[++i];
        } else if (strcmp(arg, "-m") ==0 || strcmp(arg, "--mode") == 0) {
            params->mode = argv[++i];
        } else if (strcmp(arg, "-y") == 0) {
            params->system_prompt = argv[++i];
        } else if (strcmp(arg, "--threads") == 0) {
            params->n_threads = atoi(argv[++i]);
        } else {
            error_usage();
            return 0;
        }
    }

    return 1;
}

void show_settings(GptParams params) {
  printf("---------------------SETTINGS----------------------\n");
  printf("  seed: %lld\n", params.seed);
  printf("  n_predict: %d\n", params.n_predict);
  printf("  temperature: %.2f\n", params.temperature);
  printf("  topp: %.2f\n", params.topp);
  printf("  checkpoint_path: %s\n", params.checkpoint_path);
  printf("  tokenizer_path: %s\n", params.tokenizer_path);
  printf("  prompt: %s\n", params.prompt);
  printf("  mode: %s\n", params.mode);
  printf("  threads: %d\n", params.n_threads);
  printf("---------------------------------------------------\n\n");
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// utilities: cpu
/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#endif
    return cpu_get_num_physical_cores();
}

int32_t cpu_get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32)
    //TODO: Implement
#endif
    // c++ implementation
    // unsigned int n_threads = std::thread::hardware_concurrency();
    unsigned int n_threads = sysconf(_SC_NPROCESSORS_ONLN); // number of processors currently online
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

void set_numa_thread_affinity(int n_threads) {
    // TODO
}

void clear_numa_thread_affinity() {
    // TODO
}