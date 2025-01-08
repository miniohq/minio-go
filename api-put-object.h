// Copyright 2024 - MinIO, Inc. All rights reserved.

#ifndef _PUT_OBJECT_H
#define _PUT_OBJECT_H

#include <unistd.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char *access_key;
  char *secret_key;
  char *session_token;
} Credentials;

typedef void* cClient;

void *Aligned(size_t n, int c);
void *AlignedGPU(size_t n, int c);
void FreeGPU(void *ptr);
void GPUToHost(void *dst, void *src, size_t size);
  
char *PutObjectRDMA(cClient clnt, const char *bucket, const char *object, void *buf, size_t size);
char *GetObjectRDMA(cClient clnt, const char *bucket, const char *object, void *buf, size_t size);
  
cClient newMinioCPP(char *endpoint, bool secure, char *region, Credentials creds);

#ifdef __cplusplus
}
#endif

#endif // _PUT_OBJECT_H
