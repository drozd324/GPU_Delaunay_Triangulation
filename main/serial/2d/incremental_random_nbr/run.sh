#!/bin/bash
cd "${0%/*}" || exit # Run from this directory

#make clean
rm ./data/data.txt
rm ./data/plots/*.png

make
./bin/test $@
python3 plot.py
