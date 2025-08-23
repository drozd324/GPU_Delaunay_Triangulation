#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory
GPUDIR="../../gpu"
cd $GPUDIR

MINSIZE=$((2 ** 2))
MAXSIZE=$((2 ** 20))
STEP=2

MAXS=2

NDISTRIBITIONS=4

NTPB=128 # number of threads per block

EXEDIR="./bin/test"
DATADIR="data/coredata.csv"
PLOTDATA_DOUBLE="../plotting/floatVsDouble/dataDouble.csv"
PLOTDATA_FLOAT="../plotting/floatVsDouble/dataFloat.csv"
PLOTDIR="../plotting/floatVsDouble"
> "$PLOTDATA_DOUBLE"
> "$PLOTDATA_FLOAT"

# FOR DOUBLE PRECISION
# replace '//#define NOFLIP' with '#define NOFLIP'
#sed 's/\/\/#define NOFLIP/#define NOFLIP/' ./src/types.h
sed -i 's/#define REAL float/\/\/#define REAL float/' ./src/types.h
sed -i 's/#define REALFLOAT/\/\/#define REALFLOAT/' ./src/types.h
sed -i 's/\/\/\#define REAL double/#define REAL double/' ./src/types.h
sed -i 's/\/\/\#define REALDOUBLE/#define REALDOUBLE/' ./src/types.h
make clean
make

wait

echo "$EXEDIR" -n "$N"
RUN=$( { "$EXEDIR" -n "$N" ;} )
head -n 1 "$DATADIR" >> "$PLOTDATA_DOUBLE"


for (( s=0; s<$MAXS; s++)); do
	for (( n=$MINSIZE; n<=$MAXSIZE; n*=$STEP)); do
		for (( d=0; d<$NDISTRIBITIONS; d++)); do
			echo     "$EXEDIR" -n "$n" -s "$s" -d "$d" -t "$NTPB"
			RUN=$( { "$EXEDIR" -n "$n" -s "$s" -d "$d" -t "$NTPB" ; } )

			tail -n 1 "$DATADIR" >> "$PLOTDATA_DOUBLE"
		done
	done
done



# FOR SINGLE PRECISION
#sed -i 's/\/\/#define NOFLIP/#define NOFLIP/' ./src/types.h
sed -i 's/\/\/#define REAL float/#define REAL float/' ./src/types.h
sed -i 's/\/\/#define REALFLOAT/#define REALFLOAT/' ./src/types.h
sed -i 's/#define REAL double/\/\/#define REAL double/' ./src/types.h
sed -i 's/#define REALDOUBLE/\/\/#define REALDOUBLE/' ./src/types.h
make clean
make

wait

echo "$EXEDIR" -n "$N"
RUN=$( { "$EXEDIR" -n "$N" ;} )
head -n 1 "$DATADIR" >> "$PLOTDATA_FLOAT"

for (( s=0; s<$MAXS; s++)); do
	for (( n=$MINSIZE; n<=$MAXSIZE; n*=$STEP)); do
		for (( d=0; d<$NDISTRIBITIONS; d++)); do
			echo     "$EXEDIR" -n "$n" -s "$s" -d "$d" -t "$NTPB"
			RUN=$( { "$EXEDIR" -n "$n" -s "$s" -d "$d" -t "$NTPB" ; } )

			tail -n 1 "$DATADIR" >> "$PLOTDATA_FLOAT"
		done
	done
done

make clean

wait

PLOTDIR="../plotting/floatVsDouble"
cd $PLOTDIR
$HOME/.venv/bin/python3 plot.py

