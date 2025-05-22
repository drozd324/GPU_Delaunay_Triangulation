execs = assignment6a\
		assignment6b\
		assignment6c\

CXX :=  g++ 
CXXFLAGS := -Wextra -Wall --std=c++23
DEBUG = -g -fsanitize=address -lefence

all: $(execs)

assignment6a: assignment6a.cc 
	$(CXX) $(CXXFLAGS) -o $@ $^ $(DEBUG)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -f *.o $(execs)
