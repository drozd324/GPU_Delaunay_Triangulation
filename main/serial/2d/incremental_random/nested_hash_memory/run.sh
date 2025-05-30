#!/bin/bash
cd "${0%/*}" || exit # Run from this directory

make
./main

rm ./data/triangles_iterations_plots/*.png
python3 plot.py	
