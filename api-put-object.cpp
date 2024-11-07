// Copyright 2024 - MinIO, Inc. All rights reserved.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>

#include <miniocpp/args.h>
#include <miniocpp/client.h>
#include <miniocpp/providers.h>
#include <miniocpp/request.h>
#include <miniocpp/response.h>
#include "api-put-object.h"

void *Aligned(size_t bufsize, int c) {
  char *bufptr;
  int res = posix_memalign((void **)&bufptr, getpagesize(), bufsize);
  if (res) {
    return NULL;
  }
  assert(bufptr);
  memset(bufptr, 'B', bufsize);
  return bufptr;
}

void *AlignedGPU(size_t bufsize, int c) {
  char *bufptr;
  
  cudaMalloc(&bufptr, bufsize);
  cudaMemset(bufptr,'B', bufsize);
  cudaStreamSynchronize(0);
  return bufptr;
}

void FreeGPU(void *bufptr) {
  cudaFree((char *) bufptr);
}

void GPUToHost(void *dst, void *src, size_t size) {
  cudaMemcpy((char *)dst, (char *)src, size, cudaMemcpyDeviceToHost);
}

int PutObjectRDMA(cClient clnt, const char *bucket, const char *object, void *buf, size_t size) {
  minio::s3::Client* ret = (minio::s3::Client *) clnt;

  minio::s3::PutObjectRDMAArgs *args = new minio::s3::PutObjectRDMAArgs;
  args->buf = ((char *) buf);
  args->size = size;
  args->bucket = bucket;
  args->object = object;

  minio::s3::PutObjectResponse resp = ret->PutObject(*args);
  if (resp) {
    return 1;
  }
  return -1;
}

int GetObjectRDMA(cClient clnt, const char *bucket, const char *object, void *buf, size_t size) {
  minio::s3::Client* ret = (minio::s3::Client *) clnt;

  minio::s3::GetObjectRDMAArgs *args = new minio::s3::GetObjectRDMAArgs;
  args->buf = ((char *) buf);
  args->size = size;
  args->bucket = bucket;
  args->object = object;

  minio::s3::GetObjectResponse resp = ret->GetObject(*args);
  if (resp) {
    return 1;
  }
  return -1;
}

cClient newMinioCPP(char *endpoint, bool secure, char *region, Credentials creds) {
  // Create S3 base URL.
  minio::s3::BaseUrl base_url(endpoint, secure, region);

  // Create credential provider.
  minio::creds::StaticProvider *provider = new minio::creds::StaticProvider(creds.access_key, creds.secret_key, creds.session_token);

  minio::s3::Client *ret = new minio::s3::Client(base_url, provider);

  return (void *) ret;
}
