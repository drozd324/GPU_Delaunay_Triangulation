#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd "$GPUDIR" # need to run gpu code in this directory

VID=make 

STARTN=100
MAXN=100000 # max num of points
STEP=1000 
MAXS=10 # max seeds

NDISTRIBITIONS=4

NTPB=128 # number of threads per block

#STZ="/usr/local/cuda-12.8/bin/compute-sanitizer"

EXEDIR="./bin/test"
DATADIR="data/coredata.csv"
PLOTDATA="../plotting/nptsVsTime/data.csv"
> "$PLOTDATA"

#"$STZ" "$EXEDIR" -n "$STARTN" -s 0 -d 0 -t "$NTPB"
echo "$EXEDIR" -n "$STARTN" -s 0 -d 0 -t "$NTPB"
RUN=$( { "$EXEDIR" -n "$STARTN" -s 0 -d 0 -t "$NTPB"; } )
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

PLOTDIR="../plotting/nptsVsTime"
cd $PLOTDIR
$HOME/.venv/bin/python3 plot.py
