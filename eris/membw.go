package main
import (
	"log"
)

const (
	limitFull = 100
)

type MemBw struct {
	AbsResource
	curLimit		int
	minBandwidth	int
	bandwidthGranularity int
}

func (m *MemBw) update() {
	if m.isFullLevel() {
		m.curLimit = limitFull
	} else if m.isMinLevel() {
		m.curLimit = m.minBandwidth
	} else {
		m.curLimit = m.minBandwidth + m.level * m.bandwidthGranularity
	}
}

func (m *MemBw) budgeting(bes []*Container, lcs []*Container) {
	m.update()

	setMBA(bestEffortCOS, uint(m.curLimit))
	log.Printf("set best-efforts COS memory bandwidth to %v", m.curLimit)
}