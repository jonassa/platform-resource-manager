package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"os"
)

type BinThreshold struct {
	UtilStart float64 `json:"util_start"`
	UtilEnd   float64 `json:"util_end"`
	Cpi       float64 `json:"cpi"`
	Mpki      float64 `json:"mpki"`
	Mb        float64 `json:"mb"`
	Cpl3m     float64 `json:"cpl3m"`
	Mspki     float64 `json:"mspki"`
	PMM       float64 `json:"pmm"`
}

type WorkloadThreshold struct {
	TdpThreshold *struct {
		Util float64 `json:"util"`
		Mean float64 `json:"mean"`
		Std  float64 `json:"std"`
		Bar  float64 `json:"bar"`
	} `json:"tdp_threshold"`
	MetricsThreshold []BinThreshold `json:"metrics_threshold"`
}

type Threshold struct {
	LcUtilMax float64                      `json:"lcutilmax"`
	Workloads map[string]WorkloadThreshold `json:"workloads"`
}

var thresholds = Threshold{}

type HeatmapBin struct {
	UtilStart  float64
	UtilEnd    float64
	LLCHeatmap map[string]float64
	MBHeatmap  map[string]float64
	PMMHeatmap map[string]float64
}

var heatmaps = make(map[string][]*HeatmapBin)

func initHeatmaps() {
	for name, v := range thresholds.Workloads {
		binThresholds := v.MetricsThreshold
		for _, threshold := range binThresholds {
			var heatmap HeatmapBin
			heatmap.UtilStart = threshold.UtilStart
			heatmap.UtilEnd = threshold.UtilEnd
			heatmap.LLCHeatmap = make(map[string]float64)
			heatmap.MBHeatmap = make(map[string]float64)
			heatmap.PMMHeatmap = make(map[string]float64)
			heatmaps[name] = append(heatmaps[name], &heatmap)
		}
	}
}

func printHeatmap() {
	for k, v := range heatmaps {
		log.Printf(k)
		for _, bin := range v {
			llcMapString, _ := json.Marshal(bin.LLCHeatmap)
			mbMapString, _ := json.Marshal(bin.MBHeatmap)
			pmmMapString, _ := json.Marshal(bin.PMMHeatmap)
			log.Printf("%f %f llc: %s mb: %s pmm: %s", bin.UtilStart, bin.UtilEnd, llcMapString, mbMapString, pmmMapString)
		}
	}
}

func initThreshold() {
	f, err := os.OpenFile(*threshFile, os.O_RDONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	bs, err := ioutil.ReadAll(f)
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(bs, &thresholds)
	if err != nil {
		panic(err)
	}
	initHeatmaps()
}

func updateLcUtilMax(lcmax float64) {
	thresholds.LcUtilMax = lcmax
	if *detect {
		thresh, err := json.Marshal(thresholds)
		if err != nil {
			panic(err)
		}
		err = ioutil.WriteFile(*threshFile, thresh, 0644)
		if err != nil {
			panic(err)
		}
	}
	if *control {
		cpuq.updateSysMaxUtil(lcmax)
	}
}

func detectTDPContention(m Metric, cm map[int]bool) {
	if t, ok := thresholds.Workloads[m.Name]; ok && t.TdpThreshold != nil {
		if m.CPUUtilization >= t.TdpThreshold.Util && float64(m.NormalizedFrequency) < t.TdpThreshold.Bar {
			cm[tdpContention] = true
			log.Printf("TDP Contention ! Workload: %s, CPU Usage: %+v, Frequency: %+v, Thresh: %+v, Bar: %+v\n",
				m.Name, m.CPUUtilization, m.NormalizedFrequency, t.TdpThreshold.Util, t.TdpThreshold.Bar)
		}
	}
}

func detectContenderHeuristic(metrics map[string]Metric, id string, ct int) {
	suspect := "unknown"
	maxValue := 0.0

	for n, m := range metrics {
		var val float64
		if n != id {
			if ct == llcContention {
				val = float64(m.CacheOccupancy)
			} else if ct == mbwContention {
				val = m.MemoryBandwidthTotal
			} else if ct == pmmContention {
				val = m.PMMInstPercentage
			}
			if val > maxValue {
				maxValue = val
				suspect = n
			}
		}
	}
	log.Printf("Suspect Contender: %s\n", suspect)
}

func detectContender(metrics map[string]Metric, id string, ct int, util float64) {
	suspect := "unknown"
	resource := "unknown"
	maxDelta := 0.0
	heatmapBins := heatmaps[id]

	var heatmap map[string]float64

	for _, bin := range heatmapBins {
		if util >= bin.UtilStart && util <= bin.UtilEnd {
			if ct == llcContention {
				heatmap = bin.LLCHeatmap
				resource = "cache occupancy"
			} else if ct == mbwContention {
				heatmap = bin.MBHeatmap
				resource = "memory bandwidth"
			} else if ct == pmmContention {
				heatmap = bin.PMMHeatmap
				resource = "persistent memory"
			}
		}
	}

	if heatmap == nil {
		detectContenderHeuristic(metrics, id, ct)
		return
	}

	for n, m := range metrics {
		var delta float64
		if n != id {
			historyData := 0.0
			if val, ok := heatmap[m.Name]; ok {
				historyData = float64(val)
			}

			if ct == llcContention {
				delta = float64(m.CacheOccupancy) - historyData
			} else if ct == mbwContention {
				delta = m.MemoryBandwidthTotal - historyData
			} else if ct == pmmContention {
				delta = m.PMMInstPercentage - historyData
			}

			if delta > maxDelta {
				maxDelta = delta
				suspect = n
			}
		}
	}

	if suspect != "unknown" {
		log.Printf("Suspect Contender: %s, %s increased: %f\n", suspect, resource, maxDelta)
		return
	}
	detectContenderHeuristic(metrics, id, ct)
}

func detectInBin(bt BinThreshold, m Metric, metrics map[string]Metric, cm map[int]bool) {
	if m.CyclesPerInstruction > bt.Cpi {
		unknown := true
		if m.CacheMissPerKiloInstructions > bt.Mpki {
			cm[llcContention] = true
			unknown = false
			log.Printf("LLC Contention detected on %s, CPU Usage: %+v, CPI: %+v, Thresh: %+v, MPKI: %+v, Thresh: %+v\n",
				m.Name, m.CPUUtilization, m.CyclesPerInstruction, bt.Cpi, m.CacheMissPerKiloInstructions, bt.Mpki)
			detectContender(metrics, m.Name, llcContention, m.CPUUtilization)
		}
		if m.CyclesPerL3Miss > 0 && m.CyclesPerL3Miss > bt.Cpl3m {
			cm[mbwContention] = true
			unknown = false
			log.Printf("MB Contention detected on %s, CPU Usage: %+v, CPI: %+v, Thresh: %+v, CPL3M: %+v, Thresh: %+v\n",
				m.Name, m.CPUUtilization, m.CyclesPerInstruction, bt.Cpi, m.CyclesPerL3Miss, bt.Cpl3m)

			if m.PMMInstruction > 0 && m.PMMInstPercentage < bt.PMM {
				cm[pmmContention] = true
				log.Printf("AEP Contention on %s, PMM Percentage: %+v, Thresh: %+v\n", m.Name, m.PMMInstPercentage, bt.PMM)
			}
			detectContender(metrics, m.Name, mbwContention, m.CPUUtilization)
		} else if m.StallsMemoryLoadPerKiloInstructions > 0 && m.StallsMemoryLoadPerKiloInstructions > bt.Mspki && unknown {
			// detect memory contention with mspki only if memory latency metrics is not available and cache contention is not identified
			cm[mbwContention] = true
			unknown = false
			log.Printf("MB Contention detected on %s, CPU Usage: %+v, CPI: %+v, Thresh: %+v, MSPKI: %+v, Thresh: %+v\n",
				m.Name, m.CPUUtilization, m.CyclesPerInstruction, bt.Cpi, m.StallsMemoryLoadPerKiloInstructions, bt.Mspki)
			detectContender(metrics, m.Name, mbwContention, m.CPUUtilization)
		}

		if unknown {
			log.Printf("Unknown Contention ! Workload %s, CPU Usage: %+v, CPI: %+v, Thresh: %+v\n",
				m.Name, m.CPUUtilization, m.CyclesPerInstruction, bt.Cpi)
		}
	}

	heatmapBins := heatmaps[m.Name]
	u := m.CPUUtilization
	for _, bin := range heatmapBins {
		if u > bin.UtilStart && u <= bin.UtilEnd {
			if cm[llcContention] == false {
				llcHeatmap := make(map[string]float64)
				for name, metric := range metrics {
					llcHeatmap[name] = metric.CacheOccupancy
				}
				bin.LLCHeatmap = llcHeatmap
			}
			if cm[mbwContention] == false {
				mbHeatmap := make(map[string]float64)
				for name, metric := range metrics {
					mbHeatmap[name] = metric.MemoryBandwidthTotal
				}
				bin.MBHeatmap = mbHeatmap
			}
			if cm[pmmContention] == false {
				pmmHeatmap := make(map[string]float64)
				for name, metric := range metrics {
					pmmHeatmap[name] = metric.PMMInstPercentage
				}
				bin.PMMHeatmap = pmmHeatmap
			}
		}
	}
}

func detectContention(metrics map[string]Metric, m Metric, cm map[int]bool) {
	if t, ok := thresholds.Workloads[m.Name]; ok {
		u := m.CPUUtilization

		for i := 0; i < len(t.MetricsThreshold); i++ {
			bt := t.MetricsThreshold[i]
			if u < bt.UtilStart {
				if i == 0 {
					return
				}
				detectInBin(t.MetricsThreshold[i-1], m, metrics, cm)
				return
			}
			if u <= bt.UtilEnd || i == len(t.MetricsThreshold)-1 {
				detectInBin(bt, m, metrics, cm)
				return
			}
		}
	}
}
