// Copyright (C) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions
// and limitations under the License.
//
//
// SPDX-License-Identifier: Apache-2.0

package main

// #cgo CFLAGS: -fstack-protector-strong
// #cgo CFLAGS: -fPIE -fPIC
// #cgo CFLAGS: -O2 -D_FORTIFY_SOURCE=2
// #cgo CFLAGS: -Wformat -Wformat-security
// #cgo LDFLAGS: -lpqos
// #cgo LDFLAGS: -Wl,-z,noexecstack
// #cgo LDFLAGS: -Wl,-z,relro
// #cgo LDFLAGS: -Wl,-z,now
// #include <sys/sysinfo.h>
import "C"

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
)

var workloadConfFile = flag.String("workload_conf_file", "workload.json", "workload configuration file describes each task name, type, id, request cpu count")
var verbose = flag.Bool("verbose", false, "increase output verbosity")
var detect = flag.Bool("detect", false, "detect resource contention between containers")
var control = flag.Bool("control", false, "regulate best-efforts task resource usages")
var recordMetric = flag.Bool("record-metric", false, "record container platform metrics in csv file")
var recordUtil = flag.Bool("record-util", false, "record container CPU utilizaton in csv file")
var keyCid = flag.Bool("key-cid", false, "use container id in workload configuration file as key id")
var enableHold = flag.Bool("enable-hold", false, "keep container resource usage in current level while the usage is close but not exceed throttle threshold")
var disableCat = flag.Bool("disable-cat", false, "disable CAT control while in resource regulation")
var disableMba = flag.Bool("disable-mba", false, "disable MBA control while in resource regulation")
var disableQuota = flag.Bool("disable-quota", false, "disable quota control while in resource regulation")
var exclusiveCat = flag.Bool("exclusive-cat", false, "use exclusive CAT control while in resource regulation")
var prometheusPort = flag.Int("prometheus-port", 0, "allow eris send metrics to Prometheus")
var utilInterval = flag.Int("util-interval", 2, "CPU utilization monitor interval")
var metricInterval = flag.Int("metric-interval", 20, "platform metrics monitor interval")
var llcCycles = flag.Int("llc-cycles", 6, "cycle number in LLC controller")
var mbwCycles = flag.Int("mbw-cycles", 6, "cycle number in MEMBW controller")
var quotaCycles = flag.Int("quota-cycles", 7, "cycle number in CPU CFS quota controller")
var marginRatio = flag.Float64("margin-ratio", 0.5, "margin ratio related to one logical processor used in CPU cycle regulation")
var threshFile = flag.String("thresh-file", "threshold.json", "threshold model file build from analyze.py tool")
var metricFile = flag.String("metric-file", "metric.csv", "file to store collected metrics")
var utilFile = flag.String("util-file", "util.csv", "file to store collected utilization")

var numCPU int

func init() {
	numCPU = int(C.get_nprocs())
}

func main() {
	flag.Parse()
	initMetric()
	initPerf()
	newDockerClient()
	readCgroupDriver()
	initWorkload()

	if *prometheusPort != 0 {
		go prometheusStart([]interface{}{Metric{}, Utilization{}})
	}

	if *detect {
		initThreshold()
	}
	if *control {
		initController()
	}

	go handleData()
	go startCollectMetrics()

	c := make(chan os.Signal)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	<-c
	log.Printf("exiting...\n")
	if metricCsvWriter != nil && metricCsvFile != nil {
		metricCsvWriter.Flush()
		metricCsvFile.Sync()
		metricCsvFile.Close()
	}
	if utilCsvWriter != nil && utilCsvFile != nil {
		utilCsvWriter.Flush()
		utilCsvFile.Sync()
		utilCsvFile.Close()
	}
	pqosFin()
}
