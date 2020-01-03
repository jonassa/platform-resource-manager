package main

import (
	"encoding/csv"
	"os"
)

var metricChannel = make(chan map[string]Metric, 2)
var utilizationChannel = make(chan []Utilization, 1000)

func initCsvHeader(csvFile string, v interface{}) (*csv.Writer, *os.File) {
	var csvWriter *csv.Writer
	f, err := os.OpenFile(csvFile, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		f, err = os.OpenFile(csvFile, os.O_CREATE|os.O_APPEND|os.O_RDWR, 0644)
		if err != nil {
			panic(err)
		}
		csvWriter = csv.NewWriter(f)
		csvWriter.Write(getHeaders(v))
		csvWriter.Flush()
	} else {
		csvWriter = csv.NewWriter(f)
	}
	return csvWriter, f
}

var metricCsvWriter, utilCsvWriter *csv.Writer
var metricCsvFile, utilCsvFile *os.File

func handleData() {

	if *recordMetric {
		metricCsvWriter, metricCsvFile = initCsvHeader(*metricFile, Metric{})
	}
	if *recordUtil {
		utilCsvWriter, utilCsvFile = initCsvHeader(*utilFile, Utilization{})
	}

	count := 0
	for {
		select {
		case utils := <-utilizationChannel:
			var lcUtils, beUtils float64
			bes := []*Container{}
			lcs := []*Container{}
			ts := utils[0].Time
			for _, u := range utils {
				if *prometheusPort != 0 {
					updateMetrics(u, u.Cid, u.Name)
				}
				if *recordUtil {
					utilCsvWriter.Write(getEntry(u))
				}

				con, exist := containers[u.Cid]
				if _, ok := bestEffort[u.Name]; ok {
					beUtils += u.CPUUtilization
					if exist {
						bes = append(bes, con)
					}
				} else {
					lcUtils += u.CPUUtilization
					if exist {
						lcs = append(lcs, con)
					}
				}
			}
			if *recordUtil {
				ulc := Utilization{Time: ts, Cid: "", Name: "lcs", CPUUtilization: lcUtils}
				utilCsvWriter.Write(getEntry(ulc))
				if count++; count%20 == 0 {
					utilCsvWriter.Flush()
				}
			}
			if lcUtils > thresholds.LcUtilMax {
				updateLcUtilMax(lcUtils)
			}
			if *control && !*disableQuota && len(bes) > 0 {
				exceed, hold := cpuq.detectMarginExceed(lcUtils, beUtils)
				if !*enableHold {
					hold = false
				}
				controllers[cycleContention].update(bes, lcs, exceed, hold)
			}
		case metrics := <-metricChannel:
			contends := map[int]bool{llcContention: false, mbwContention: false, tdpContention: false}
			for _, m := range metrics {
				if *prometheusPort != 0 {
					updateMetrics(m, m.Cid, m.Name)
				}
				if *recordMetric {
					metricCsvWriter.Write(getEntry(m))
				}
				if *detect {
					detectContention(metrics, m, contends)
					detectTDPContention(m, contends)
				}
			}
			if *control {
				for c, v:= range contends{
					controller, ok := controllers[c]
					if ok {
						controller.update(nil, nil, v, false)
					}
				}		
			}
			if *recordMetric {
				metricCsvWriter.Flush()
			}
		}
	}
}
