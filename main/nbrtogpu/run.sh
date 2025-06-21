#!/bin/bash
cd "${0%/*}" || exit # run from this directory

make clean
mkdir -p ./data
rm ./data/data.txt

make
./bin/test $@
python3 plot.py
