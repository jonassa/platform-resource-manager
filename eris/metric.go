package main

import (
	"log"
	"reflect"
	"strings"
	"time"
)

var eventIndex = map[string]int{}

type Metric struct {
	Time uint64 `header:"time"`
	Cid  string `header:"cid"`
	Name string `header:"name"`

	// Fixed Counter
	Instruction    uint64 `header:"instruction" event:"INST_RETIRED.ANY_P" gauge:"cma_instructions" gauge_help:"Instructions of a container"`
	Cycle          uint64 `header:"cycle" event:"CPU_CLK_UNHALTED.THREAD_P" gauge:"cma_unhalted_cycles" gauge_help:"Unhalted cycles of a container"`
	ReferenceCycle uint64 `header:"ref_cycle" event:"CPU_CLK_UNHALTED.REF_TSC" gauge:"cma_ref_unhalted_cycles" gauge_help:"Reference Unhalted cycles of a container"`

	// Programmable Counter
	CacheMiss        uint64 `header:"cache_miss" event:"LONGEST_LAT_CACHE.MISS" gauge:"cma_llc_miss" gauge_help:"Cache misses of a container"`
	L3MissRequests   uint64 `header:"l3_miss_requests" event:"OFFCORE_REQUESTS.L3_MISS_DEMAND_DATA_RD" platform:"SKX,CLX" gauge:"cma_l3miss_requests" gauge_help:"l3 miss requests count"`
	L3MissCycles     uint64 `header:"l3_miss_cycles" event:"OFFCORE_REQUESTS_OUTSTANDING.L3_MISS_DEMAND_DATA_RD" platform:"SKX,CLX" gauge:"cma_l3miss_cycles" gauge_help:"l3 miss cycle count"`
	StallsMemoryLoad uint64 `header:"stalls_memory_load" event:"CYCLE_ACTIVITY.STALLS_MEM_ANY" gauge:"cma_stalls_mem_load" gauge_help:"excution stalls while memory subsystem has an outstanding load"`
	PMMInstruction   uint64 `enabled:"false" header:"pmm_instruction" event:"MEM_LOAD_RETIRED.LOCAL_PMM" platform:"CLX" gauge:"cma_pmm_instruction" gauge_help:"instruction retired for pmm"`

	// RDT
	CacheOccupancy        float64 `header:"cache_occupancy" gauge:"cma_llc_occupancy" gauge_help:"Last level cache occupancy of a container"`
	MemoryBandwidthLocal  float64 `header:"memory_bandwidth_local"`
	MemoryBandwidthRemote float64 `header:"memory_bandwidth_remote"`
	MemoryBandwidthTotal  float64 `gauge:"cma_memory_bandwidth" gauge_help:"Total memory bandwidth of a container"`

	// CPU Utilization
	CPUUtilization float64 `header:"cpu_utilization" gauge:"cma_cpu_usage_percentage" gauge_help:"CPU usage percentage of a container"`

	// Calculation
	NormalizedFrequency                 uint64  `header:"normalized_frequency" gauge:"cma_average_frequency" gauge_help:"Normalized Frequency of a container"`
	CyclesPerInstruction                float64 `header:"cycles_per_instruction" gauge:"cma_cycles_per_instruction" gauge_help:" Cycles per instruction of a container"`
	ReferenceCyclesPerInstruction       float64 `header:"ref_cycles_per_instruction" gauge:"cma_ref_cycles_per_instruction" gauge_help:" Reference Cycles per instruction of a container"`
	CacheMissPerKiloInstructions        float64 `header:"cache_miss_per_kilo_instruction" gauge:"cma_misses_per_kilo_instruction" gauge_help:"Misses per kilo instruction of a container"`
	StallsMemoryLoadPerKiloInstructions float64 `header:"stalls_memory_load_per_kilo_instruction" gauge:"cma_stalls_mem_per_instruction" gauge_help:"Stalls memory load per instruction of a container"`
	CyclesPerL3Miss                     float64 `header:"cycles_per_l3_miss" platform:"SKX,CLX" gauge:"cma_cycles_per_l3_miss" gauge_help:"cycles per l3 miss"`
	PMMInstTotal                        uint64  `header:"pmm_inst_local_total" platform:"CLX" gauge:"pmm_inst_retired_local_total" gauge_help:"memory local instruction retired on local pmm total"`
	PMMInstPercentage                   float64 `header:"pmm_inst_retired_local_percentage" platform:"CLX" gauge:"pmm_inst_retired_local_percentage" gauge_help:"percentage of pmm inst retired local"`
}

func initMetric() {
	m := Metric{}
	mType := reflect.TypeOf(m)
	for i := 0; i < mType.NumField(); i++ {
		tags := mType.Field(i).Tag
		e := tags.Get("event")
		pf := tags.Get("platform")
		enabled := tags.Get("enabled")
		if e != "" && enabled != "false" && (pf == "" || strings.Index(pf, platform) != -1) {
			eventIndex[e] = i
		}
	}
	log.Printf("%+v\n", eventIndex)
}

var containers = map[string]*Container{}

func updateContainers() {
	cons, err := getContainers()
	if err != nil {
		log.Println(err)
	} else {
		for id, container := range containers {
			// remove all finished containers
			if _, ok := cons[id]; err == nil && !ok {
				container.finalize()
				delete(containers, id)
			}
		}

		newbe := false
		bes := []*Container{}
		lcs := []*Container{}
		for id, name := range cons {
			name = strings.TrimLeft(name, "/")
			con, ok := containers[id]
			if !ok {
				// initialize new containter
				cgroup, err := newContainer(id, name)
				if err == nil {
					containers[id] = cgroup
					if *control {
						if _, ok := bestEffort[name]; ok {
							newbe = true
							bes = append(bes, cgroup)
							setShare(cgroup, shareBe)
						} else {
							lcs = append(lcs, cgroup)
							setShare(cgroup, shareLc)
						}
					}
				}
			} else {
				if *control {
					if _, ok := bestEffort[name]; ok {
						bes = append(bes, con)
					} else {
						lcs = append(lcs, con)
					}
				}
			}
		}
		if *control && !*disableQuota && newbe {
			cpuq.budgeting(bes, lcs)
		}
	}
}

func startCollectMetrics() {
	ticker := newDelayedTicker(0, time.Duration(*metricInterval)*time.Second)
	utilTicker := newDelayedTicker(0, time.Duration(*utilInterval)*time.Second)
	pqosUpdate := time.NewTicker(time.Duration(1 * time.Second))
	cacheOccupancySample := newDelayedTicker(2, 500*time.Millisecond)
	for {
		select {
		case <-cacheOccupancySample.C:
			for _, c := range containers {
				if c.monitorStarted {
					// read pqos cache occupancy data
					if llc, err := c.pollCacheOccupancy(); err == nil {
						c.cacheOccupancyCount++
						c.cacheOccupancySum += llc
					}
				}
			}
		case <-pqosUpdate.C:
			for _, c := range containers {
				pidsMap, err := listTaskPid(c.id)
				if err != nil {
					log.Printf("%+v", err)
					continue
				} else {
					var err error
					if c.isLatencyCritical {
						err = updatePqosGroup(c.pqosMonitorData, latencyCriticalCOS, c.pqosPidsMap, pidsMap)
					} else if c.isBestEffort {
						err = updatePqosGroup(c.pqosMonitorData, bestEffortCOS, c.pqosPidsMap, pidsMap)
					} else {
						err = updatePqosGroup(c.pqosMonitorData, genericCOS, c.pqosPidsMap, pidsMap)
					}
					if err != nil {
						log.Println(err)
					}
					c.pqosPidsMap = pidsMap
				}

			}
		case <-utilTicker.C:
			ts := uint64(time.Now().Unix())
			updateContainers()
			var utils []Utilization
			for id, container := range containers {
				u := Utilization{Time: ts, Cid: id, Name: container.name}
				cpuData := container.pollCPUUsage(false)
				if cpuData != nil && cpuData[1] != 0 {
					u.CPUUtilization = float64(cpuData[0]) / float64(cpuData[1]) * float64(numCPU) * 100.0
					utils = append(utils, u)
				}
			}
			if len(utils) > 0 {
				utilizationChannel <- utils
			}
		case <-ticker.C:
			ts := uint64(time.Now().Unix())
			updateContainers()
			metrics := map[string]Metric{}
			var pmmRetiredInstTotalData uint64 = 0
			for id, container := range containers {
				if !container.monitorStarted {
					container.start()
					container.monitorStarted = true
					continue
				}

				m := Metric{Time: ts, Cid: id, Name: container.name}
				// read perf data
				perfData := container.pollPerf()
				if perfData != nil {
					for i := 0; i < len(peCounters); i++ {
						m.setEventMetric(peCounters[i].EventName, perfData[i])
					}
				}
				// read cpu utilization data
				cpuData := container.pollCPUUsage(true)
				if cpuData != nil && cpuData[1] != 0 {
					m.CPUUtilization = float64(cpuData[0]) / float64(cpuData[1]) * float64(numCPU) * 100.0
				}

				// read pqos memory bandwidth data
				memoryBandwidthData := container.pollMemoryBandwidth()
				if memoryBandwidthData != nil {
					m.MemoryBandwidthLocal = float64(memoryBandwidthData[1]) / 1024.0 / 1024.0 / float64(*metricInterval)
					m.MemoryBandwidthRemote = float64(memoryBandwidthData[2]) / 1024.0 / 1024.0 / float64(*metricInterval)
				}

				var llcValid bool
				if container.cacheOccupancyCount > 0 {
					llcValid = true
					m.CacheOccupancy = float64(container.cacheOccupancySum) / float64(container.cacheOccupancyCount)
					container.cacheOccupancySum, container.cacheOccupancyCount = 0, 0
				}

				m.calculate()
				if perfData != nil && cpuData != nil && memoryBandwidthData != nil && llcValid {
					metrics[container.name] = m
				}

				pmmRetiredInstTotalData += m.PMMInstruction
			}

			if len(metrics) > 0 {
				metricsFinal := map[string]Metric{}
				for k, v := range metrics {
					v.PMMInstTotal = pmmRetiredInstTotalData
					if pmmRetiredInstTotalData != 0 {
						v.PMMInstPercentage = float64(v.PMMInstruction) / float64(v.PMMInstTotal) * 100
					} else {
						v.PMMInstPercentage = 0.0
					}
					metricsFinal[k] = v
				}
				metricChannel <- metricsFinal
			}
		}
	}
}

func (m *Metric) setEventMetric(event string, value uint64) {
	index, ok := eventIndex[event]
	if !ok {
		panic("trying to set a not predefined value")
	}
	mPtr := reflect.ValueOf(m)
	mPtr.Elem().Field(index).SetUint(value)
}

func (m *Metric) calculate() {
	if m.Instruction != 0 {
		m.CyclesPerInstruction = float64(m.Cycle) / float64(m.Instruction)
		m.ReferenceCyclesPerInstruction = float64(m.ReferenceCycle) / float64(m.Instruction)
		m.CacheMissPerKiloInstructions = float64(m.CacheMiss) / float64(m.Instruction) * 1000.0
		m.StallsMemoryLoadPerKiloInstructions = float64(m.StallsMemoryLoad) / float64(m.Instruction) * 1000
	}
	if m.CPUUtilization != 0 {
		m.NormalizedFrequency = uint64(float64(m.Cycle) / float64(*metricInterval) / 10000.0 / m.CPUUtilization)
	}
	if m.L3MissRequests != 0 {
		m.CyclesPerL3Miss = float64(m.L3MissCycles) / float64(m.L3MissRequests)
	}
	m.MemoryBandwidthTotal = m.MemoryBandwidthLocal + m.MemoryBandwidthRemote
}
