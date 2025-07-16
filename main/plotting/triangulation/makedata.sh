#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd "$GPUDIR" # need to run gpu code in this directory

N=100 # max num of points
SEED=69420
DISRIBUTION=1
NTPB=128 # number of threads per block

#STZ="/usr/local/cuda-12.8/bin/compute-sanitizer"

EXEDIR="./bin/test"
DATADIR="data/coredata.csv"
PLOTDATA="../plotting/triangulation/tri.txt"
> "$PLOTDATA"

echo "$EXEDIR" -n "$N" -s "$SEED" -d "$DISRIBUTION" -t "$NTPB"
RUN=$( { "$EXEDIR" -n "$N" -s "$SEED" -d "$DISRIBUTION" -t "$NTPB" ; } )

cp data/tri.txt "$PLOTDATA"
