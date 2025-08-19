#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd "$GPUDIR" # need to run gpu code in this directory

VOID=make 

N=$((10 ** 5)) # max num of points
MAXS=1 # max seeds
NTPB=128

NDISTRIBITIONS=1

#STZ="/usr/local/cuda-12.8/bin/compute-sanitizer"

EXEDIR="./bin/test"
DATADIR="data/flipedPerIter.txt"
PLOTDATA="../plotting/nflipsVsIter/flipedPerIter.txt"
> "$PLOTDATA"

echo     "$EXEDIR" -n "$N"
RUN=$( { "$EXEDIR" -n "$N"; } )

for (( s=0; s<$MAXS; s++)); do
	for (( d=0 ; d<$NDISTRIBITIONS; d++)); do
		echo "$EXEDIR" -n "$N" -s "$s" -d "$d" -t "$NTPB"
		RUN=$( { "$EXEDIR" -n "$N" -s "$s" -d "$d" -t "$NTPB" ; } )

		cp "$DATADIR" "$PLOTDATA"
	done
done
