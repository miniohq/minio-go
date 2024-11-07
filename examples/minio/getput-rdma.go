//go:build example
// +build example

package main

import (
	"context"
	"log"
	"os"
	"strconv"
	"unsafe"

	"github.com/dustin/go-humanize"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

func main() {
	metaClient, err := minio.NewC(os.Args[1], &minio.Options{
		Creds:  credentials.NewStaticV4(os.Args[2], os.Args[3], ""),
		Secure: false,
	})
	if err != nil {
		log.Fatalln(err)
	}

	size := humanize.MiByte
	gpuEnabled := false
	if len(os.Args) >= 5 {
		size, err = strconv.Atoi(os.Args[4])
		if err != nil {
			log.Fatalln(err)
		}
		if len(os.Args) == 6 {
			gpuEnabled = true
		}
	}

	var buf unsafe.Pointer
	if gpuEnabled {
		buf = minio.AlignedGPU(size, 'G')
		defer minio.FreeGPU(buf)
	} else {
		buf = minio.Aligned(size, 'C')
		defer minio.Free(buf)
	}

	info, err := metaClient.PutObjectRDMA(context.Background(), "my-bucket", "my-object-go", buf, size, minio.PutObjectOptions{})
	if err != nil {
		log.Fatalln(err)
	}
	log.Println(info, "Uploaded successfully")

	err = metaClient.GetObjectRDMA(context.Background(), "my-bucket", "my-object-go", buf, size, minio.GetObjectOptions{})
	if err != nil {
		log.Fatalln(err)
	}

	hostbuf := minio.Aligned(size, 'X')
	defer minio.Free(hostbuf)

	if gpuEnabled {
		minio.GPUToHost(hostbuf, buf, size)
	} else {
		minio.HostToHost(hostbuf, buf, size)
	}

	if err = os.WriteFile("output.txt", (*[minio.MaxArrayLen]byte)(hostbuf)[:size:size], 0o644); err != nil {
		log.Fatalln(err)
	}

	log.Println("Downloaded", "my-object-go", "of size: ", size, "Successfully.")
}
