#!/bin/bash
cd "${0%/*}" || exit 1  # Run from script's directory

make
mkdir -p ./data

SEED=7437891
MAXN=10000000
STEP=10
FILE_NAME="./data/time.csv"
echo "n,time" > "$FILE_NAME"
echo "n,time"

for (( j=10; j<$MAXN; j*=$STEP )); do
    TIME=$( { time -p ./bin/test -n "$j" -s "$SEED"; } 2>&1 | sed -n 's/^real //p' )
    echo "$j,$TIME" >> "$FILE_NAME"
    echo "$j,$TIME"
done
