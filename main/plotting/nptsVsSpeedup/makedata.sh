#!/bin/bash

cd "${0%/*}" || exit 1  # Run from script's directory

STARTN=10
MAXN=50 # max num of points
STEP=10 
MAXS=2 # max seeds
NDISTRIBITIONS=2

./makedataCPU.sh $STARTN $MAXN $STEP $MAXS $NDISTRIBITIONS
./makedataGPU.sh $STARTN $MAXN $STEP $MAXS $NDISTRIBITIONS

