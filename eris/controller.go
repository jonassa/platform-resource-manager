package main

type Controller interface {
	update(bes []*Container, lcs []*Container, detected bool, hold bool)
}

var cpuq CpuQuota
var controllers = map[int]Controller{}

func initController() {
	cpuq = CpuQuota{AbsResource: AbsResource{level: resLevMin, levelMax: resLevMax}}
	cpuq.updateSysMaxUtil(thresholds.LcUtilMax)
	quotaController := NaiveController{res: &cpuq, cycThresh: *quotaCycles}
	controllers[cycleContention] = &quotaController

	if !*disableCat {
		llcMaxLevel := l3way - 1
		if *exclusiveCat {
			llcMaxLevel = l3way / 2
		}
		llc := LlcOccupy{AbsResource: AbsResource{level: resLevMin, levelMax: llcMaxLevel} }
		llc.initBitmap()
		llc.budgeting(nil, nil)
		llcController := NaiveController{res: &llc, cycThresh: *llcCycles}
		controllers[llcContention] = &llcController
	}

	if !*disableMba {
		mbMaxLevel := int((limitFull - minBandwidth) / bandwidthGranularity)
		mb := MemBw{AbsResource: AbsResource{level: resLevMin, levelMax: mbMaxLevel},
				minBandwidth: minBandwidth, bandwidthGranularity: bandwidthGranularity}
		mb.budgeting(nil, nil)
		mbController := NaiveController{res: &mb, cycThresh: *mbwCycles}
		controllers[mbwContention] = &mbController
	}
}
