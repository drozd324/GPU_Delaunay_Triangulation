#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd "$GPUDIR" # need to run gpu code in this directory

VOID=make 

STARTN=10
N=100 # max num of points
MAXS=1 # max seeds
NTPB=128

NDISTRIBITIONS=1

#STZ="/usr/local/cuda-12.8/bin/compute-sanitizer"

EXEDIR="./bin/test"
DATADIR="data/insertedPerIter.txt"
PLOTDATA="../plotting/ninsertVsIter/insertedPerIter.txt"
> "$PLOTDATA"

for (( s=0; s<$MAXS; s++)); do
	for (( d=0 ; d<$NDISTRIBITIONS; d++)); do
		echo "$EXEDIR" -n "$N" -s "$s" -d "$d" -t "$NTPB"
		RUN=$( { "$EXEDIR" -n "$N" -s "$s" -d "$d" -t "$NTPB" ; } )

		cp "$DATADIR" "$PLOTDATA"
	done
done
