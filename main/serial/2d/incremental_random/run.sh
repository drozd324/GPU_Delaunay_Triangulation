#!/bin/bash
cd "${0%/*}" || exit # Run from this directory

make clean
rm ./data/triangles_iterations_plots/*.png
rm ./data/triangles_iterations/*.txt
rm ./data/points.txt

make
./bin/main
python3 plot.py	
