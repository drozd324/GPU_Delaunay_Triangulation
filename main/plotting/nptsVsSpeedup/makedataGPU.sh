#!/bin/bash

# ========================= GRAB GPU DATA ========================= 

cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd "$GPUDIR" # need to run gpu code in this directory

make 

STARTN=$1
MAXN=$2 # max num of points
STEP=$3 
MAXS=$4 # max seeds
NDISTRIBITIONS=$5

NTPB=128 # number of threads per block

#STZ="/usr/local/cuda-12.8/bin/compute-sanitizer"

EXEDIR="./bin/test"
DATADIR="data/coredata.csv"
PLOTDATA="../plotting/nptsVsSpeedup/dataGPU.csv"
> "$PLOTDATA"

#"$STZ" "$EXEDIR" -n "$STARTN" -s 0 -d 0 -t "$NTPB"
echo "$EXEDIR" -n "$STARTN"
RUN=$( { "$EXEDIR" -n "$STARTN"; } )
HEAD=$(head -n 1 "$DATADIR")
echo "$HEAD" >> "$PLOTDATA"

# for each size
for (( s=0; s<$MAXS; s++)); do
	for (( n=$STARTN; n<=$MAXN; n+=$STEP)); do
		for (( d=0 ; d<$NDISTRIBITIONS; d++)); do
			echo "$EXEDIR" -n "$n" -s "$s" -d "$d" -t "$NTPB"
			RUN=$( { "$EXEDIR" -n "$n" -s "$s" -d "$d" -t "$NTPB" ; } )
			DATA=$(tail -n 1 "$DATADIR")
			echo "$DATA" >> "$PLOTDATA"
		done
	done
done
