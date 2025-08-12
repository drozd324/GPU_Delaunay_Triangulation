#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
CPUDIR="../../serialIncPtInsertion"
cd "$CPUDIR" # need to run gpu code in this directory

VOID=make 

N=1000 # max num of points
SEED=69420
DISRIBUTION=1

#STZ="/usr/local/cuda-12.8/bin/compute-sanitizer"

EXEDIR="./bin/test"
DATADIR="data/coredata.csv"
PLOTDATA="../plotting/serial_triangulation/tri.txt"
> "$PLOTDATA"

echo "$EXEDIR" -n "$N" -s "$SEED" -d "$DISRIBUTION"
RUN=$( { "$EXEDIR" -n "$N" -s "$SEED" -d "$DISRIBUTION"; } )

cp data/tri.txt "$PLOTDATA"
