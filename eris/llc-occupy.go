package main
import (
	"log"
)

type LlcOccupy struct {
	AbsResource
	beBmp	[]uint64
	lcBmp	[]uint64
}

func (llc *LlcOccupy) initBitmap() {
	if !catSupported {
		log.Printf("CAT is not supported or enabled!")
		return
	}
	llc.beBmp = make([]uint64, l3way - 1)
	llc.lcBmp = make([]uint64, l3way - 1)
	var i uint64
	bitCnt := uint64(l3way)
	for i = 1; i < bitCnt; i++{
		llc.beBmp[i - 1] = ((1 << (i + 1)) - 1) << (bitCnt - 1 - i) 
		llc.lcBmp[i - 1] = (1 << (bitCnt - 1 - i)) - 1
	}
	if *exclusiveCat {
		llc.beBmp = llc.beBmp[:int(l3way/2)]
		llc.lcBmp = llc.lcBmp[:int(l3way/2)]
	}
	// log.Printf("beBmp is %v, lcBmp is %v", llc.beBmp, llc.lcBmp)
}

func (llc *LlcOccupy) budgeting(bes []*Container, lcs []*Container) {
	level := llc.level
	if llc.isFullLevel() {
		level = llc.levelMax - 1
	}
	if *exclusiveCat {
		setCAT(latencyCriticalCOS, llc.lcBmp[level])
		log.Printf("set latency critical COS llc occupancy to %x", llc.lcBmp[level])
	}

	setCAT(bestEffortCOS, llc.beBmp[level])
	log.Printf("set best-efforts COS llc occupancy to %x", llc.beBmp[level])
}