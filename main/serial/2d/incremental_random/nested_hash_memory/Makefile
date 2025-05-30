execs = main

CXX :=  g++ 
CXXFLAGS := -Wextra -Wall --std=c++23
DEBUG = -g #-fsanitize=address -lefence 

all: $(execs)

main: main.cpp src/delaunay.h 
	$(CXX) -o $@ $^ $(DEBUG) $(CXXFLAGS) 

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) 


.PHONY: clean

clean:
	rm -f *.o $(execs)
