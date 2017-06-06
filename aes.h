#include "core.h"
#include "argon2.h"

#ifndef PORTABLE_AES_H
#define PORTABLE_AES_H

#if defined(__cplusplus)
extern "C" {
#endif

//internal stuff


enum aes_constant {
    AES_BLOCKBYTES = 128,
   	AES_OUTBYTES = 64,
    AES_KEYBYTES = 64,
    AES_SALTBYTES = 16,
    AES_PERSONALBYTES = 16
};

//macros
#define DO_ENC_BLOCK(m,k) \
	do{\
        m = _mm_xor_si128       (m, k[ 0]); \
        m = _mm_aesenc_si128    (m, k[ 1]); \
        m = _mm_aesenc_si128    (m, k[ 2]); \
        m = _mm_aesenc_si128    (m, k[ 3]); \
        m = _mm_aesenclast_si128(m, k[ 4]); \
    }while(0)

#define DO_DEC_BLOCK(m,k) \
	do{\
        m = _mm_xor_si128       (m, k[4+0]); \
        m = _mm_aesdec_si128    (m, k[4+1]); \
        m = _mm_aesdec_si128    (m, k[4+2]); \
        m = _mm_aesdec_si128    (m, k[4+3]); \
        m = _mm_aesdeclast_si128(m, k[  0]);   \
    }while(0)

#define AES_128_key_exp(k, rcon) aes_128_key_expansion(k, _mm_aeskeygenassist_si128(k, rcon))

// static __m128i key_schedule[20];//the expanded key
static __m128i aes_128_key_expansion(__m128i key, __m128i keygened);

//public API
void aes128_load_key(int8_t *enc_key, __m128i *key_schedule);

void aes128_enc(__m128i *key_schedule,int8_t *plainText,int8_t *cipherText);

void aes128_dec(__m128i *key_schedule,int8_t *cipherText,int8_t *plainText);

/* My round function/wide cipher using AES */

void AES_ROUND(__m128i *state, size_t message_len, __m128i *global_key, uint32_t local_parallelism);

