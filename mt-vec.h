#ifndef __MT_VEC_H__
#define __MT_VEC_H__

// from llama.cpp
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)
#include <arm_neon.h>

// F32 NEON

#define GGML_F32_STEP 16
#define GGML_F32_EPR  4

#define GGML_F32x4              float32x4_t
#define GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#define GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#define GGML_F32x4_LOAD         vld1q_f32
#define GGML_F32x4_STORE        vst1q_f32
#define GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define GGML_F32x4_ADD          vaddq_f32
#define GGML_F32x4_MUL          vmulq_f32
#define GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#define GGML_F32x4_REDUCE(res, x)              \
{                                              \
    int offset = GGML_F32_ARR >> 1;            \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vaddq_f32(x[i], x[offset+i]);   \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vaddq_f32(x[i], x[offset+i]);   \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vaddq_f32(x[i], x[offset+i]);   \
    }                                          \
    res = GGML_F32x4_REDUCE_ONE(x[0]);         \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define GGML_F16_STEP 32
    #define GGML_F16_EPR  8

    #define GGML_F16x8              float16x8_t
    #define GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #define GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #define GGML_F16x8_LOAD(x)      vld1q_f16((const ggml_fp16_internal_t *)(x))
    #define GGML_F16x8_STORE        vst1q_f16
    #define GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #define GGML_F16x8_ADD          vaddq_f16
    #define GGML_F16x8_MUL          vmulq_f16
    #define GGML_F16x8_REDUCE(res, x)                             \
    do {                                                          \
        int offset = GGML_F16_ARR >> 1;                           \
        for (int i = 0; i < offset; ++i) {                        \
            x[i] = vaddq_f16(x[i], x[offset+i]);                  \
        }                                                         \
        offset >>= 1;                                             \
        for (int i = 0; i < offset; ++i) {                        \
            x[i] = vaddq_f16(x[i], x[offset+i]);                  \
        }                                                         \
        offset >>= 1;                                             \
        for (int i = 0; i < offset; ++i) {                        \
            x[i] = vaddq_f16(x[i], x[offset+i]);                  \
        }                                                         \
        const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 (x[0])); \
        const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(x[0])); \
        res = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } while (0)

    #define GGML_F16_VEC                GGML_F16x8
    #define GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #define GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #define GGML_F16_VEC_LOAD(p, i)     GGML_F16x8_LOAD(p)
    #define GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((ggml_fp16_internal_t *)(p), r[i])
    #define GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #define GGML_F16_VEC_ADD            GGML_F16x8_ADD
    #define GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #define GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#else
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #define GGML_F16_STEP 16
    #define GGML_F16_EPR  4

    #define GGML_F32Cx4              float32x4_t
    #define GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #define GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #define GGML_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16((const ggml_fp16_internal_t *)(x)))
    #define GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #define GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #define GGML_F32Cx4_ADD          vaddq_f32
    #define GGML_F32Cx4_MUL          vmulq_f32
    #define GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #define GGML_F16_VEC                GGML_F32Cx4
    #define GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #define GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx4_LOAD(p)
    #define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((ggml_fp16_internal_t *)(p), r[i])
    #define GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #define GGML_F16_VEC_ADD            GGML_F32Cx4_ADD
    #define GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #define GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#endif

#elif defined(__AVX512F__)
#include <immintrin.h>

// F32 AVX512

#define GGML_F32_STEP 64
#define GGML_F32_EPR  16

#define GGML_F32x16         __m512
#define GGML_F32x16_ZERO    _mm512_setzero_ps()
#define GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#define GGML_F32x16_LOAD    _mm512_loadu_ps
#define GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#define GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define GGML_F32x16_ADD     _mm512_add_ps
#define GGML_F32x16_MUL     _mm512_mul_ps
#define GGML_F32x16_REDUCE(res, x)                                    \
do {                                                                  \
    int offset = GGML_F32_ARR >> 1;                                   \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    res = _mm512_reduce_add_ps(x[0]);                                 \
} while (0)

#define GGML_F32_VEC        GGML_F32x16
#define GGML_F32_VEC_ZERO   GGML_F32x16_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x16_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x16_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x16_STORE
#define GGML_F32_VEC_FMA    GGML_F32x16_FMA
#define GGML_F32_VEC_ADD    GGML_F32x16_ADD
#define GGML_F32_VEC_MUL    GGML_F32x16_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512

// F16 AVX

#define GGML_F16_STEP 64
#define GGML_F16_EPR  16

// AVX512 has FP16 extension (AVX512_FP16) but I don't have it on my machine so I use FP32 instead

#define GGML_F32Cx16             __m512
#define GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#define GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#define GGML_F32Cx16_LOAD(x)     _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x)))
#define GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#define GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define GGML_F32Cx16_ADD         _mm512_add_ps
#define GGML_F32Cx16_MUL         _mm512_mul_ps
#define GGML_F32Cx16_REDUCE(res, x)                               \
do {                                                              \
    int offset = GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    res = _mm512_reduce_add_ps(x[0]);                             \
} while (0)

#define GGML_F16_VEC                GGML_F32Cx16
#define GGML_F16_VEC_ZERO           GGML_F32Cx16_ZERO
#define GGML_F16_VEC_SET1           GGML_F32Cx16_SET1
#define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx16_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#define GGML_F16_VEC_FMA            GGML_F32Cx16_FMA
#define GGML_F16_VEC_ADD            GGML_F32Cx16_ADD
#define GGML_F16_VEC_MUL            GGML_F32Cx16_MUL
#define GGML_F16_VEC_REDUCE         GGML_F32Cx16_REDUCE

#endif 

#define GGML_F32_ARR (GGML_F32_STEP / GGML_F32_EPR)
#define GGML_F16_ARR (GGML_F16_STEP / GGML_F16_EPR)

#endif