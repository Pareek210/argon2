#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h>     //for int8_t
#include <wmmintrin.h>  //for intrinsics for AES-NI
#include <smmintrin.h>

#include "core.h"
#include "aes.h"
//compile using gcc and following arguments: -g;-O0;-Wall;-msse2;-msse;-march=native;-maes

#define blockAES __m128i

/* ------------------------------------------------------------------------- */

#define zero           _mm_setzero_si128()
#define vadd(x,y)      _mm_add_epi8(x,y)
#define vand(x,y)      _mm_and_si128(x,y)
#define vandnot(x,y)   _mm_andnot_si128(x,y)  /* (~x)&y */
#define vor(x,y)       _mm_or_si128(x,y)
#define vxor(x,y)      _mm_xor_si128(x,y)

static int is_zero(blockAES x) { return _mm_testz_si128(x,x); }      /* 0 or 1 */

static blockAES sll4(blockAES x) {
    return vor(_mm_srli_epi64(x, 4), _mm_slli_epi64(_mm_srli_si128(x, 8), 60));
}

static blockAES srl4(blockAES x) {
    return vor(_mm_slli_epi64(x, 4), _mm_srli_epi64(_mm_slli_si128(x, 8), 60));
}

static __m128i aes4(__m128i in, __m128i a, __m128i b,
                    __m128i c, __m128i d, __m128i e) {
    in = _mm_aesenc_si128(vxor(in,a),b);
    in = _mm_aesenc_si128(in,c);
    in = _mm_aesenc_si128(in,d);
    return _mm_aesenc_si128 (in,e);
}

static __m128i loadu(const void *p) { return _mm_loadu_si128((__m128i*)p); }
static void storeu(const void *p, __m128i x) {_mm_storeu_si128((__m128i*)p,x);}

#define load loadu      /* Intel with AES-NI has fast unaligned loads/stores */
#define store storeu

#define vxor3(x,y,z)        vxor(vxor(x,y),z)
#define vxor4(w,x,y,z)      vxor(vxor(w,x),vxor(y,z))
#define load_partial(p,n)   loadu(p)

// static __m128i key_schedule[20];//the expanded key

static __m128i aes_128_key_expansion(__m128i key, __m128i keygened){
	keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3,3,3,3));
	key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	return _mm_xor_si128(key, keygened);
}

//public API
void aes128_load_key(int8_t *enc_key, __m128i *key_schedule){
    key_schedule[0] = _mm_loadu_si128((const __m128i*) enc_key);
	key_schedule[1]  = AES_128_key_exp(key_schedule[0], 0x01);
	key_schedule[2]  = AES_128_key_exp(key_schedule[1], 0x02);
	key_schedule[3]  = AES_128_key_exp(key_schedule[2], 0x04);
	key_schedule[4]  = AES_128_key_exp(key_schedule[3], 0x08);

	// generate decryption keys in reverse order.
    // k[4] is shared by last encryption and first decryption rounds
    // k[0] is shared by first encryption round and last decryption round (and is the original user key)
    // For some implementation reasons, decryption key schedule is NOT the encryption key schedule in reverse order
	key_schedule[5] = _mm_aesimc_si128(key_schedule[3]);
	key_schedule[6] = _mm_aesimc_si128(key_schedule[2]);
	key_schedule[7] = _mm_aesimc_si128(key_schedule[1]);
}

void aes128_enc(__m128i *key_schedule,int8_t *plainText,int8_t *cipherText){
    __m128i m = _mm_loadu_si128((__m128i *) plainText);

    DO_ENC_BLOCK(m,key_schedule);

    _mm_storeu_si128((__m128i *) cipherText, m);
}

void aes128_dec(__m128i *key_schedule,int8_t *cipherText,int8_t *plainText){
    __m128i m = _mm_loadu_si128((__m128i *) cipherText);

    DO_DEC_BLOCK(m,key_schedule);

    _mm_storeu_si128((__m128i *) plainText, m);
}


static blockAES pass_one(blockAES *global_key, blockAES *src, unsigned blocks, uint32_t local_parallelism) {
    blockAES I=global_key[0];
    blockAES J=global_key[1];
    blockAES k=global_key[2];
    blockAES L=global_key[3];
    unsigned bytes = blocks * 16 ;
    while (bytes >= 16*16) {
        store(src+ 0, aes4(load(src + 1),I, J, K, L, load(src+ 0)));
        store(src+ 2, aes4(load(src + 3),I, J, K, L, load(src+ 2)));
        store(src+ 4, aes4(load(src + 5),I, J, K, L, load(src+ 4)));
        store(src+ 6, aes4(load(src + 7),I, J, K, L, load(src+ 6)));
        store(src+ 8, aes4(load(src + 9),I, J, K, L, load(src+ 8)));
        store(src+10, aes4(load(src +11),I, J, K, L, load(src+10)));
        store(src+12, aes4(load(src +13),I, J, K, L, load(src+12)));
        store(src+14, aes4(load(src +15),I, J, K, L, load(src+14)));
        tmp=aes4(I,load(src+ 0),J,K,L,load(src+ 1));store(src+ 1,tmp);
        sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+ 2),J,K,L,load(src+ 3));
        store(src+ 3,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+ 4),J,K,L,load(src+ 5));
        store(src+ 5,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+ 6),J,K,L,load(src+ 7));
        store(src+ 7,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+ 8),J,K,L,load(src+ 9));
        store(src+ 9,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+10),J,K,L,load(src+11));
        store(src+11,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+12),J,K,L,load(src+13));
        store(src+13,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+14),J,K,L,load(src+15));
        store(src+15,tmp);sum=vxor(sum,tmp);
        bytes -= 16*16; src += 16;
    }
    if (bytes >= 8*16) {
        store(src+ 0, aes4(load(src + 1),I, J, K, L, load(src+ 0)));
        store(src+ 2, aes4(load(src + 3),I, J, K, L, load(src+ 2)));
        store(src+ 4, aes4(load(src + 5),I, J, K, L, load(src+ 4)));
        store(src+ 6, aes4(load(src + 7),I, J, K, L, load(src+ 6)));
        tmp=aes4(I,load(src+ 0),J,K,L,load(src+ 1));
        store(src+ 1,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+ 2),J,K,L,load(src+ 3));
        store(src+ 3,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+ 4),J,K,L,load(src+ 5));
        store(src+ 5,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+ 6),J,K,L,load(src+ 7));
        store(src+ 7,tmp);sum=vxor(sum,tmp);
        bytes -= 8*16; src += 8;
    }
    if (bytes >= 4*16) {
        store(src+ 0, aes4(load(src + 1),I, J, K, L, load(src+ 0)));
        store(src+ 2, aes4(load(src + 3),I, J, K, L, load(src+ 2)));
        tmp=aes4(I,load(src+ 0),J,K,L,load(src+ 1));
        store(src+ 1,tmp);sum=vxor(sum,tmp);
        tmp=aes4(I,load(src+ 2),J,K,L,load(src+ 3));
        store(src+ 3,tmp);sum=vxor(sum,tmp);
        bytes -= 4*16; src += 4; 
    }
    if (bytes) {
        store(src+ 0, aes4(load(src + 1),I, J, K, L, load(src+ 0)));
        tmp=aes4(I,load(src+ 0),J,K,L,load(src+ 1));
        store(src+ 1,tmp);sum=vxor(sum,tmp);
    }
    return sum;
}


/* ------------------------------------------------------------------------- */


static blockAES pass_two(blockAES *global_key, blockAES s, blockAES *src, unsigned blocks, uint32_t local_parallelism) {
    blockAES I=global_key[0];
    blockAES J=global_key[1];
    blockAES k=global_key[2];
    blockAES L=global_key[3];
    unsigned bytes = blocks * 16 ;
    blockAES fs[8];

    /* Discuss this */
    while (bytes >= 16*16) {
        fs[0] = aes4pre(s,I,J,K,L); fs[1] = aes4pre(s,I,J,K,L);
        fs[2] = aes4pre(s,I,J,K,L); fs[3] = aes4pre(s,I,J,K,L);
        fs[4] = aes4pre(s,I,J,K,L); fs[5] = aes4pre(s,I,J,K,L);
        fs[6] = aes4pre(s,I,J,K,L); fs[7] = aes4pre(s,I,J,K,L);
        tmp[0] = vxor(load(src+ 0),fs[0]); sum = vxor(sum,tmp[0]);
        store(src+ 0,vxor(load(src+ 1),fs[0]));
        tmp[1] = vxor(load(src+ 2),fs[1]); sum = vxor(sum,tmp[1]);
        store(src+ 2,vxor(load(src+ 3),fs[1]));
        tmp[2] = vxor(load(src+ 4),fs[2]); sum = vxor(sum,tmp[2]);
        store(src+ 4,vxor(load(src+ 5),fs[2]));
        tmp[3] = vxor(load(src+ 6),fs[3]); sum = vxor(sum,tmp[3]);
        store(src+ 6,vxor(load(src+ 7),fs[3]));
        tmp[4] = vxor(load(src+ 8),fs[4]); sum = vxor(sum,tmp[4]);
        store(src+ 8,vxor(load(src+ 9),fs[4]));
        tmp[5] = vxor(load(src+10),fs[5]); sum = vxor(sum,tmp[5]);
        store(src+10,vxor(load(src+11),fs[5]));
        tmp[6] = vxor(load(src+12),fs[6]); sum = vxor(sum,tmp[6]);
        store(src+12,vxor(load(src+13),fs[6]));
        tmp[7] = vxor(load(src+14),fs[7]); sum = vxor(sum,tmp[7]);
        store(src+14,vxor(load(src+15),fs[7]));
        store(src+ 1, aes4(I,load(src+ 0), J, I, L, tmp[0]));
        store(src+ 3, aes4(I,load(src+ 2), J, I, L, tmp[1]));
        store(src+ 5, aes4(I,load(src+ 4), J, I, L, tmp[2]));
        store(src+ 7, aes4(I,load(src+ 6), J, I, L, tmp[3]));
        store(src+ 9, aes4(I,load(src+ 8), J, I, L, tmp[4]));
        store(src+11, aes4(I,load(src+10), J, I, L, tmp[5]));
        store(src+13, aes4(I,load(src+12), J, I, L, tmp[6]));
        store(src+15, aes4(I,load(src+14), J, I, L, tmp[7]));
        store(src+ 0, aes4(load(src+ 1),o1, J, I, L, load(src+ 0)));
        store(src+ 2, aes4(load(src+ 3),o2, J, I, L, load(src+ 2)));
        store(src+ 4, aes4(load(src+ 5),o3, J, I, L, load(src+ 4)));
        store(src+ 6, aes4(load(src+ 7),o4, J, I, L, load(src+ 6)));
        store(src+ 8, aes4(load(src+ 9),o5, J, I, L, load(src+ 8)));
        store(src+10, aes4(load(src+11),o6, J, I, L, load(src+10)));
        store(src+12, aes4(load(src+13),o7, J, I, L, load(src+12)));
        store(src+14, aes4(load(src+15),o8, J, I, L, load(src+14)));
        bytes -= 16*16; src += 16;
    }
    if (bytes >= 8*16) {
        o1 = vxor(offset,L);
        o2 = vxor(offset,L2);
        o3 = vxor(o1,L2);
        o4 = offset = vxor(offset,L4);
        fs[0] = aes4pre(s,o1,J,I,L); fs[1] = aes4pre(s,o2,J,I,L);
        fs[2] = aes4pre(s,o3,J,I,L); fs[3] = aes4pre(s,o4,J,I,L);
        o1 = vxor(J3,o1); o2 = vxor(J3,o2);
        o3 = vxor(J3,o3); o4 = vxor(J3,o4);
        tmp[0] = vxor(load(src+ 0),fs[0]); sum = vxor(sum,tmp[0]);
        store(src+ 0,vxor(load(src+ 1),fs[0]));
        tmp[1] = vxor(load(src+ 2),fs[1]); sum = vxor(sum,tmp[1]);
        store(src+ 2,vxor(load(src+ 3),fs[1]));
        tmp[2] = vxor(load(src+ 4),fs[2]); sum = vxor(sum,tmp[2]);
        store(src+ 4,vxor(load(src+ 5),fs[2]));
        tmp[3] = vxor(load(src+ 6),fs[3]); sum = vxor(sum,tmp[3]);
        store(src+ 6,vxor(load(src+ 7),fs[3]));
        store(src+ 1, aes4(I,load(src+ 0), J, I, L, tmp[0]));
        store(src+ 3, aes4(I,load(src+ 2), J, I, L, tmp[1]));
        store(src+ 5, aes4(I,load(src+ 4), J, I, L, tmp[2]));
        store(src+ 7, aes4(I,load(src+ 6), J, I, L, tmp[3]));
        store(src+ 0, aes4(load(src+ 1),o1, J, I, L, load(src+ 0)));
        store(src+ 2, aes4(load(src+ 3),o2, J, I, L, load(src+ 2)));
        store(src+ 4, aes4(load(src+ 5),o3, J, I, L, load(src+ 4)));
        store(src+ 6, aes4(load(src+ 7),o4, J, I, L, load(src+ 6)));
        bytes -= 8*16; src += 8;
    }
    if (bytes >= 4*16) {
        o1 = vxor(offset,L);
        o2 = offset = vxor(offset,L2);
        fs[0] = aes4pre(s,o1,J,I,L); fs[1] = aes4pre(s,o2,J,I,L);
        o1 = vxor(J3,o1); o2 = vxor(J3,o2);
        tmp[0] = vxor(load(src+ 0),fs[0]); sum = vxor(sum,tmp[0]);
        store(src+ 0,vxor(load(src+ 1),fs[0]));
        tmp[1] = vxor(load(src+ 2),fs[1]); sum = vxor(sum,tmp[1]);
        store(src+ 2,vxor(load(src+ 3),fs[1]));
        store(src+ 1, aes4(I,load(src+ 0), J, I, L, tmp[0]));
        store(src+ 3, aes4(I,load(src+ 2), J, I, L, tmp[1]));
        store(src+ 0, aes4(load(src+ 1),o1, J, I, L, load(src+ 0)));
        store(src+ 2, aes4(load(src+ 3),o2, J, I, L, load(src+ 2)));
        bytes -= 4*16; src += 4;
    }
    if (bytes) {
        o1 = vxor(offset,L);
        fs[0] = aes4pre(s,o1,J,I,L);
        o1 = vxor(J3,o1);
        tmp[0] = vxor(load(src+ 0),fs[0]); sum = vxor(sum,tmp[0]);
        store(src+ 0,vxor(load(src+ 1),fs[0]));
        store(src+ 1, aes4(I,load(src+ 0), J, I, L, tmp[0]));
        store(src+ 0, aes4(load(src+ 1),o1, J, I, L, load(src+ 0)));
    }
    return sum;
} 


/* ------------------------------------------------------------------------- */
/*
	The one that matters. Read this and understand this.
	Inputs ->
	1. ctx - Context
	2. t - Takes into account the Additional Data
	3. d - Probably specifies whether cipher is same size as message
	4. src - Message
	5. bytes - Message length
	6. abytes - ABYTE
	7. src - Cipher
*/


void AES_ROUND(blockAES *state, size_t message_len, blockAES *global_key, uint32_t local_parallelism){

  // Compute x and store intermediate results 
    x = pass_one(global_key, state, message_len-2, local_parallelism);

    // Calculate s and final block values (y xor'd to final1 later)
    final0 = vxor(loadu(src + (bytes - 32)), x);
    final1 = loadu(src+(bytes-32)+16);
    final0 = aes4(final1, I, J, K, L, final0);
    final1 = vxor(final1, aes4pre(final0, I, J, K, L));
    s = vxor(final0, final1);
    final0 = vxor(final0, aes4pre(final1, I, J, K, L));

    y = pass_two(ctx, s, (blockAES*)src, message_len-2, local_parallelism);

    storeu(src + (bytes - 32), vxor(final1, y));
    storeu(src + (bytes - 32) + 16, final0);

}

























