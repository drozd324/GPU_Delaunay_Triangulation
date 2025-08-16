#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory

# This bash script should make all of the plots included in the report by 
# running the GPU and CPU executables, gathering relavant data in each directory
# and saving images of plots genereated by the python scripts


#nflipsVsIter  ninsertVsIter   triangulation
DIRS=("blockSizeVsTime" "nptsVsTime" "timeDistrib" "serial_nptsVsTime" "nptsVsSpeedup" "triangulation_grid" "triangulation_history" "triangulation_onlyptins" "floatVsDouble")
for dir in "${DIRS[@]}"; do
	cd $dir
	pwd
	./makedata.sh

	cd ..
done
