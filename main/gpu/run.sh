#!/bin/bash
cd "${0%/*}" || exit # run from this directory

#make clean
mkdir -p ./data
rm ./data/data.txt

make
./bin/test $@ #> out 2>&1
#python3 plot.py
