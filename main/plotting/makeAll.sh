#!/bin/bash

#nflipsVsIter  ninsertVsIter   triangulation
DIRS=("blockSizeVsTime" "nptsVsTime" "timeDistrib" "serial_nptsVsTime" "nptsVsSpeedup")
for dir in "${DIRS[@]}"; do
	cd $dir
	pwd
	./makedata.sh

	$HOME/.venv/bin/python3 plot.py 

	#git add "$dir.png"

	cd ..
done
