#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd "$GPUDIR" # need to run gpu code in this directory

N=100 # max num of points
MAXS=5 # max seeds
NDISTRIBITIONS=3
NTPB=128


#STZ="/usr/local/cuda-12.8/bin/compute-sanitizer"

EXEDIR="./bin/test"
DATADIR="data/coredata.csv"
PLOTDATA="../plotting/dataDistrib/data.csv"
> "$PLOTDATA"

echo "$EXEDIR" -n "$N"
RUN=$( { "$EXEDIR" -n "$N" ;} )
head -n 1 "$DATADIR" >> "$PLOTDATA"

# for each size
for (( s=0; s<$MAXS; s++)); do
	for (( d=0 ; d<$NDISTRIBITIONS; d++)); do
		echo "$EXEDIR" -n "$N" -s "$s" -d "$d" -t "$t"
		RUN=$( { "$EXEDIR" -n "$N" -s "$s" -d "$d" -t "$NTPB" ; } )

		tail -n 1 "$DATADIR" >> "$PLOTDATA"
	done
done
