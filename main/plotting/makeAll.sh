#!/bin/bash

#nflipsVsIter  ninsertVsIter   triangulation
DIRS=("blocksizeVsTime" "nptsVsTime" "timeDistrib" "serial_nptsVsTime" "nptsVsSpeedup")
for dir in "${DIRS[@]}"; do
	cd $dir
	pwd
	./makedata.sh
	python3 plot.py 
	cd ..
done
