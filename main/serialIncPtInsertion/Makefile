CXX := g++
CXXFLAGS := -O3 -Wall -Wextra -std=c++20
DEBUG := #-g -fsanitize=address,undefined   #-lefence

OBJDIR := obj
BINDIR := bin
SRCDIR := src

EXEC_NAMES := test
EXECS := $(addprefix $(BINDIR)/, $(EXEC_NAMES))

CPP_SOURCES := test.cpp mymath.cpp point.cpp circle.cpp tri.cpp delaunay.cpp ran.cpp
ALL_OBJS := $(addprefix $(OBJDIR)/, $(patsubst %.cpp, %.o, $(CPP_SOURCES)))

all: $(EXECS)

$(BINDIR)/test: $(ALL_OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(DEBUG) -o $@ $^ -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(DEBUG) -c $< -o $@ 

$(OBJDIR)/tri.o: $(SRCDIR)/tri.cpp $(OBJDIR)/point.o $(OBJDIR)/circle.o
$(OBJDIR)/delaunay.o: $(SRCDIR)/delaunay.cpp $(OBJDIR)/mymath.o $(OBJDIR)/point.o $(OBJDIR)/circle.o $(OBJDIR)/tri.o

$(OBJDIR) $(BINDIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(EXECS) ./data/data.txt
