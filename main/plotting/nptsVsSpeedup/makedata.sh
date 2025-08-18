#!/bin/bash

cd "${0%/*}" || exit 1  # Run from script's directory

STARTN=100
MAXN=100000 # max num of points
STEP=100 
MAXS=10 # max seeds
NDISTRIBITIONS=4

./makedataGPU.sh $STARTN $MAXN $STEP $MAXS $NDISTRIBITIONS
./makedataCPU.sh $STARTN $MAXN $STEP $MAXS $NDISTRIBITIONS

cd "${0%/*}" || exit 1 
pwd
$HOME/.venv/bin/python3 plot.py

