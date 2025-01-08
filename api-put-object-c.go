// Copyright 2024 - MinIO, Inc. All rights reserved.

package minio

// #cgo CXXFLAGS: -I/usr/local/cuda/include --std=c++17 -g -O0
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda  -lcudart -lrt -L/usr/local/lib -lminiocpp
// #include <stdlib.h>
// #include <string.h>
// #include "api-put-object.h"
import "C"

import (
	"context"
	"errors"
	"unsafe"

	"github.com/minio/minio-go/v7/pkg/credentials"
)

// The go:linkname directives provides backdoor access to private functions in
// the runtime. Below we're accessing the throw function.

//go:linkname throw runtime.throw
func throw(s string)

func convertToCCreds(creds credentials.Value) C.Credentials {
	return C.Credentials{
		access_key:    C.CString(creds.AccessKeyID),
		secret_key:    C.CString(creds.SecretAccessKey),
		session_token: C.CString(creds.SessionToken),
	}
}

type MetaClient struct {
	GoClient *Client
	CClient  C.cClient
}

func NewC(endpoint string, opts *Options) (*MetaClient, error) {
	clnt, err := New(endpoint, opts)
	if err != nil {
		return nil, err
	}

	value, err := opts.Creds.Get()
	if err != nil {
		return nil, err
	}

	// Create S3 client.
	return &MetaClient{
		GoClient: clnt,
		CClient: C.newMinioCPP(
			C.CString(clnt.EndpointURL().String()),
			C.bool(opts.Secure),
			C.CString(opts.Region),
			convertToCCreds(value),
		),
	}, nil
}

const (
	// MaxArrayLen is a safe maximum length for slices on this architecture.
	MaxArrayLen = 1<<50 - 1
)

// Aligned returns aligned memory of n bytes
func Aligned(n int, c byte) unsafe.Pointer {
	ptr := C.Aligned(C.size_t(n), C.int(c))
	if ptr == nil {
		throw("out of memory")
	}

	return unsafe.Pointer(ptr)
}

// Free frees the specified slice.
func Free(ptr unsafe.Pointer) {
	if ptr == nil {
		throw("invalid pointer")
	}

	C.free(ptr)
}

func AlignedGPU(n int, c byte) unsafe.Pointer {
	ptr := C.AlignedGPU(C.size_t(n), C.int(c))
	if ptr == nil {
		throw("out of memory")
	}

	return unsafe.Pointer(ptr)
}

func FreeGPU(ptr unsafe.Pointer) {
	if ptr == nil {
		throw("invalid pointer")
	}

	C.FreeGPU(ptr)
}

func GPUToHost(dstPtr unsafe.Pointer, srcPtr unsafe.Pointer, size int) {
	C.GPUToHost(dstPtr, srcPtr, C.size_t(size))
}

func HostToHost(dstPtr unsafe.Pointer, srcPtr unsafe.Pointer, size int) {
	C.memcpy(dstPtr, srcPtr, C.size_t(size))
}

func (clnt *MetaClient) GetObjectRDMA(ctx context.Context, bucketName, objectName string, buf unsafe.Pointer, size int, opts GetObjectOptions) error {
	bucketC := C.CString(bucketName)
	defer C.free(unsafe.Pointer(bucketC))

	objectC := C.CString(objectName)
	defer C.free(unsafe.Pointer(objectC))

	ret := C.GetObjectRDMA(clnt.CClient, bucketC, objectC, buf, C.size_t(size))
	if ret != nil {
		return errors.New(C.GoString(ret))
	}
	return nil
}

func (clnt *MetaClient) PutObjectRDMA(ctx context.Context, bucketName, objectName string, buf unsafe.Pointer, size int, opts PutObjectOptions) (UploadInfo, error) {
	bucketC := C.CString(bucketName)
	defer C.free(unsafe.Pointer(bucketC))

	objectC := C.CString(objectName)
	defer C.free(unsafe.Pointer(objectC))

	ret := C.PutObjectRDMA(clnt.CClient, bucketC, objectC, buf, C.size_t(size))
	if ret != nil {
		return UploadInfo{}, errors.New(C.GoString(ret))
	}

	return UploadInfo{Bucket: bucketName, Key: objectName, Size: int64(size)}, nil
}
