#!/bin/bash

#nflipsVsIter  ninsertVsIter   triangulation
DIRS=("blockSizeVsTime" "nptsVsTime" "timeDistrib" "serial_nptsVsTime" "nptsVsSpeedup")
for dir in "${DIRS[@]}"; do
	cd $dir
	pwd
	#./makedata.sh
	#git add "$dir.png"
	#echo "git add $dir.png"
	python3 plot.py 
	cd ..
done
