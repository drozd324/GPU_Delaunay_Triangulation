#!/bin/bash
cd "${0%/*}" || exit # run from this directory

#make clean

make

#/usr/local/cuda-12.8/bin/compute-sanitizer ./bin/test $@ > out 2>&1
./bin/test -n 100 -s 69 -d 1 -t 128 #> out 2>&1
#python3 plot.py
