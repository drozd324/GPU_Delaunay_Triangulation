#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd "$GPUDIR" # need to run gpu code in this directory

make 

N=100000 # num of points
MAXS=5 # max seeds
NDISTRIBITIONS=1
NTPB=128

EXEDIR="./bin/test"
DATADIR="./data/coredata.csv"

GPUMODELNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
#GPUMODELNAME=${GPUMODELNAME// /_}
echo "$GPUMODELNAME"

PLOTDATA="../plotting/gpuModelTest/data_${GPUMODELNAME}.csv"
> "$PLOTDATA"

echo "$EXEDIR" -n "$N"
RUN=$( { "$EXEDIR" -n "$N" ;} )
head -n 1 "$DATADIR" >> "$PLOTDATA"

# for each size
for (( s=0; s<$MAXS; s++)); do
	for (( d=0 ; d<$NDISTRIBITIONS; d++)); do
		echo "$EXEDIR" -n "$N" -s "$s" -d "$d" -t "$t" "$NTPB" 
		RUN=$( { "$EXEDIR" -n "$N" -s "$s" -d "$d" -t "$NTPB" ; } )

		tail -n 1 "$DATADIR" >> "$PLOTDATA"
	done
done

PLOTDIR="../plotting/gpuModelTest"
cd $PLOTDIR
$HOME/.venv/bin/python3 plot.py

