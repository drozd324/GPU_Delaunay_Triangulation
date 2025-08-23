#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"


SIZESN=("100" "500" "1000")
SEED=69420

NDISTRIBITIONS=4

NTPB=128 # number of threads per block

EXEDIR="./bin/test"
DATADIR="data/tri.txt"
PLOTDATA="../plotting/triangulation_grid/tri.txt"
PLOTDIR="../plotting/triangulation_grid"

cd "$GPUDIR"
make clean
sed -i 's/bool saveHistory = false/bool saveHistory = true/' ./src/delaunay.h
make
cd "$PLOTDIR"

#"$STZ" "$EXEDIR" -n "$STARTN" -s 0 -d 0 -t "$NTPB"

for n in "${SIZESN[@]}"; do
	for (( d=0; d<$NDISTRIBITIONS; d++)); do
		cd "$GPUDIR" # need to run gpu code in this directory
		make 

		echo
		echo     "$EXEDIR" -n "$n" -s "$SEED" -d "$d" -t "$NTPB"
		RUN=$( { "$EXEDIR" -n "$n" -s "$SEED" -d "$d" -t "$NTPB" ; } )

		cp ./data/tri.txt $PLOTDIR
		#cp ./data/tri.txt $PLOTDATA

		cd $PLOTDIR
		$HOME/.venv/bin/python3 plot.py save "tri${n}_${d}.png"
	done
done

cd "$GPUDIR"
make clean
sed -i 's/bool saveHistory = true/bool saveHistory = false/' ./src/delaunay.h
cd "$PLOTDIR"

rm tri.txt
