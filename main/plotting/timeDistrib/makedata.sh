#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd "$GPUDIR" # need to run gpu code in this directory

make 

N=10000 # num of points
MAXS=1 # max seeds
NDISTRIBITIONS=2
NTPB=128

#STZ="/usr/local/cuda-12.8/bin/compute-sanitizer"

EXEDIR="./bin/test"
DATADIR="./data/coredata.csv"
PLOTDATA="../plotting/timeDistrib/data.csv"
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

PLOTDIR="../plotting/timeDistrib"
cd $PLOTDIR
$HOME/.venv/bin/python3 plot.py

