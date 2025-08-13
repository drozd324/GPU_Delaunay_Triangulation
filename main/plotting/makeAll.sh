#!/bin/bash

#nflipsVsIter  ninsertVsIter   triangulation
DIRS=("blockSizeVsTime" "nptsVsTime" "timeDistrib" "serial_nptsVsTime" "nptsVsSpeedup" "triangulation")
for dir in "${DIRS[@]}"; do
	cd $dir
	pwd
	./makedata.sh

	cd ..
done
