#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd $GPUDIR

make clean
sed -i 's/bool saveHistory = false/bool saveHistory = true/' ./src/delaunay.h
# TURN OFF FLIPPING
sed -i 's/\/\/#define NOFLIP/#define NOFLIP/' ./src/types.h
make

SIZE=500
SEED=69420
DISTRIBUTION=0

NTPB=128 # number of threads per block

EXEDIR="./bin/test"
DATADIR="data/tri.txt"
PLOTDATA="../plotting/triangulation_onlyptins/tri.txt"
PLOTDIR="../plotting/triangulation_onlyptins"
> "$PLOTDATA"

wait

echo    "$EXEDIR" -n "$SIZE" -s "$SEED" -d "$DISTRIBUTION" -t "$NTPB"
RUN=$({ "$EXEDIR" -n "$SIZE" -s "$SEED" -d "$DISTRIBUTION" -t "$NTPB" ; })
cp $DATADIR $PLOTDATA

make clean
# TURN FLIPPING BACK ON
sed -i 's/#define NOFLIP/\/\/#define NOFLIP/' ./src/types.h
sed -i 's/bool saveHistory = true/bool saveHistory = false/' ./src/delaunay.h
wait


cd $PLOTDIR
pwd
$HOME/.venv/bin/python3 plot.py iter all

rm tri.txt
