//	go build parallel.go
//
// Grab accessKey and secretKey from minio servers and set them as env
// MINIO_ACCESS_KEY and MINIO_SECRET_KEY respectively.
//
// Proceed to run the test on all the 6 nodes.
//
//	./parallel hosts <access_key> <secret_key>
package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"fmt"
	"hash/crc32"
	"log"
	"os"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"github.com/cheggaaa/pb"
	"github.com/dustin/go-humanize"
	"github.com/minio/madmin-go/v3"
	minio "github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/olekukonko/tablewriter"
)

type Operation struct {
	Start time.Time
	End   time.Time
	Size  int64
}

func progressMonitor(bar *pb.ProgressBar, recv chan Operation) {
	durations := madmin.TimeDurations{}
	var start time.Time
	var totalObjs int
	for op := range recv {
		if start.IsZero() {
			start = op.Start
		}
		durations = append(durations, op.End.Sub(op.Start))
		totalObjs++
		bar.Add64(op.Size)
	}
	bar.Finish()
	fmt.Println()

	speed := float64(bar.Get()) / (float64(time.Since(start)) / float64(time.Second))
	measured := durations.Measure()

	data := [][]string{
		{"Avg", measured.Avg.String()},
		{"P50", measured.P50.String()},
		{"P75", measured.P75.String()},
		{"P95", measured.P95.String()},
		{"P99", measured.P99.String()},
		{"P999", measured.P999.String()},
		{"Long5p", measured.Long5p.String()},
		{"Short5p", measured.Short5p.String()},
		{"Max", measured.Max.String()},
		{"Min", measured.Min.String()},
		{"StdDev", measured.StdDev.String()},
		{"Range", measured.Range.String()},
		{"Speed", humanize.IBytes(uint64(speed)) + " /s"},
		{"TotalObjects", strconv.Itoa(totalObjs)},
		{"DataTransferred", humanize.IBytes(uint64(bar.Get()))},
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{
		"Measurement",
		"Rating",
	})
	table.SetAutoWrapText(false)
	table.SetAutoFormatHeaders(true)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetCenterSeparator("")
	table.SetColumnSeparator("")
	table.SetRowSeparator("")
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetTablePadding("\t") // pad with tabs
	table.SetNoWhiteSpace(true)
	table.AppendBulk(data) // Add Bulk Data
	table.Render()
}

func readHosts(hostFile string) []string {
	f, err := os.Open(hostFile)
	if err != nil {
		return nil
	}
	defer f.Close()

	var hosts []string
	scan := bufio.NewScanner(f)
	for scan.Scan() {
		hosts = append(hosts, scan.Text())
	}
	return hosts
}

func crcHashMod(key string, cardinality int) int {
	if cardinality <= 0 {
		return -1
	}
	keyCrc := crc32.Checksum([]byte(key), crc32.IEEETable)
	return int(keyCrc % uint32(cardinality))
}

func makeNewClients(hostFile, accessKey, secretKey string) ([]*minio.MetaClient, error) {
	hosts := readHosts(hostFile)
	metaClients := make([]*minio.MetaClient, 0, len(hosts))
	for _, host := range hosts {
		metaClient, err := minio.NewC(host, &minio.Options{
			Creds:  credentials.NewStaticV4(accessKey, secretKey, ""),
			Secure: false,
		})
		if err != nil {
			return nil, err
		}
		metaClients = append(metaClients, metaClient)
	}
	return metaClients, nil
}

func getClient(key string, clients []*minio.MetaClient) *minio.MetaClient {
	return clients[crcHashMod(key, len(clients))]
}

func main() {
	if len(os.Args) < 4 {
		log.Fatalln(os.Args[0], "<hosts_file> <access_key> <secret_key>")
	}

	clients, err := makeNewClients(os.Args[1], os.Args[2], os.Args[3])
	if err != nil {
		log.Fatalln(err)
	}

	size := humanize.MiByte
	if len(os.Args) >= 5 {
		size, err = strconv.Atoi(os.Args[4])
		if err != nil {
			log.Fatalln(err)
		}
		if size < humanize.MiByte {
			size = humanize.MiByte
		}
	}

	concurrent := 128
	if len(os.Args) >= 6 {
		concurrent, err = strconv.Atoi(os.Args[5])
		if err != nil {
			log.Fatalln(err)
		}
		if concurrent < 1 {
			concurrent = 1
		}
	}

	objectsPerWorker := 10
	if len(os.Args) >= 7 {
		objectsPerWorker, err = strconv.Atoi(os.Args[6])
		if err != nil {
			log.Fatalln(err)
		}
		if objectsPerWorker < 1 {
			objectsPerWorker = 1
		}
	}

	var wg sync.WaitGroup

	bar1 := pb.New(size * concurrent * objectsPerWorker)
	// show average speed
	bar1.ShowSpeed = true

	// convert output to readable format (like KB, MB)
	bar1.SetUnits(pb.U_BYTES)
	bar1.Start()

	recv1 := make(chan Operation, concurrent*objectsPerWorker)
	go progressMonitor(bar1, recv1)

	wg.Add(concurrent)

	opts := minio.PutObjectOptions{}
	for i := 0; i < concurrent; i++ {
		go func(i int) {
			defer wg.Done()

			buf := minio.Aligned(size)
			defer minio.Free(buf)
			rand.Read(buf)

			for j := 0; j < objectsPerWorker; j++ {
				objName := fmt.Sprintf("%d-xx-%d/testobject-obj%d-worker%d.txt", j, i, j, i)
				var op Operation
				op.Start = time.Now()
				info, perr := getClient(objName, clients).PutObjectRDMA(context.Background(), "warp-benchmark-bucket", objName,
					unsafe.Pointer(&buf[0]), len(buf), opts)
				op.End = time.Now()
				if perr != nil {
					log.Println(perr)
				}
				op.Size = info.Size
				recv1 <- op
			}
		}(i)
	}

	wg.Wait()
	close(recv1)

	bar2 := pb.New(size * concurrent * objectsPerWorker)
	// show average speed
	bar2.ShowSpeed = true

	// convert output to readable format (like KB, MB)
	bar2.SetUnits(pb.U_BYTES)
	bar2.Start()

	recv2 := make(chan Operation, concurrent*objectsPerWorker)
	go progressMonitor(bar2, recv2)

	wg.Add(concurrent)

	gopts := minio.GetObjectOptions{}
	for i := 0; i < concurrent; i++ {
		go func(i int) {
			defer wg.Done()

			buf := minio.Aligned(size)
			defer minio.Free(buf)

			for j := 0; j < objectsPerWorker; j++ {
				objName := fmt.Sprintf("%d-xx-%d/testobject-obj%d-worker%d.txt", j, i, j, i)
				var op Operation
				op.Start = time.Now()
				perr := getClient(objName, clients).GetObjectRDMA(context.Background(), "warp-benchmark-bucket", objName,
					unsafe.Pointer(&buf[0]), len(buf), gopts)
				op.End = time.Now()
				if perr != nil {
					log.Println(perr)
				}
				op.Size = int64(len(buf))
				recv2 <- op
			}
		}(i)
	}

	wg.Wait()
	close(recv2)

	time.Sleep(time.Second)
}
