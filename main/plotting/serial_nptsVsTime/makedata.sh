#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
CPUDIR="../../serialIncPtInsertion"
cd "$CPUDIR" # need to run gpu code in this directory

VOID=make 

STARTN=10
MAXN=50 # max num of points
STEP=10 
MAXS=2 # max seeds

NDISTRIBITIONS=2

NTPB=128 # number of threads per block

#STZ="/usr/local/cuda-12.8/bin/compute-sanitizer"

EXEDIR="./bin/test"
DATADIR="data/coredata.csv"
PLOTDATA="../plotting/serial_nptsVsTime/data.csv"
> "$PLOTDATA"

#"$STZ" "$EXEDIR" -n "$STARTN" -s 0 -d 0 -t "$NTPB"
echo "$EXEDIR" -n "$STARTN"
RUN=$( { "$EXEDIR" -n "$STARTN"; } )
HEAD=$(head -n 1 "$DATADIR")
echo "$HEAD" >> "$PLOTDATA"

# for each size
for (( s=0; s<$MAXS; s++)); do
	for (( n=$STARTN; n<=$MAXN; n+=$STEP)); do
		for (( d=0; d<$NDISTRIBITIONS; d++)); do
			echo "$EXEDIR" -n "$n" -s "$s" -d "$d"
			RUN=$( { "$EXEDIR" -n "$n" -s "$s" -d "$d"; } )
			DATA=$(tail -n 1 "$DATADIR")
			echo "$DATA" >> "$PLOTDATA"
		done
	done
done
