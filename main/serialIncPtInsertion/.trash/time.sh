#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory

make
mkdir -p ./data

MAXN=100 # max n
STEPN=10
STARTN=10

MAXS=3 # max seeds

HEADER="n,time,seed" 

FILE_NAME="./data/time.csv"
echo $HEADER > "$FILE_NAME"
echo $HEADER

for (( s=0; s<$MAXS; s++ )); do
	for (( n=$STARTN; n<$MAXN; n+=$STEPN )); do
		TIME=$( { time -p ./bin/test -n "$n" -s "$s"; } 2>&1 | sed -n 's/^real //p' )

		DATA="$n,$TIME,$s" 

		echo $DATA >> "$FILE_NAME"
		echo $DATA
	done
done

