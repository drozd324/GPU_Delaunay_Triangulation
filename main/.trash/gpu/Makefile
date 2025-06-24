CXX := g++
CXXFLAGS := -O1 -Wall -Wextra -std=c++23
DEBUG := -g -fsanitize=address,undefined   #-lefence

NVCC := nvcc
NVCCFLAGS := -O4 --use_fast_math --compiler-options -funroll-loops -arch=sm_75 -rdc=true -Wno-deprecated-gpu-targets 
#DEBUGNV = -g -G --target-processes
DEBUGNV = -g -G --ptxas-options=-v -Xcompiler -Wall

OBJDIR := obj
BINDIR := bin
SRCDIR := src

EXEC_NAMES := test
EXECS := $(addprefix $(BINDIR)/, $(EXEC_NAMES))

CPP_SOURCES := test.cu mymath.cu point.cu circle.cu tri.cu delaunay.cu ran.cu
ALL_OBJS := $(addprefix $(OBJDIR)/, $(patsubst %.cu, %.o, $(CPP_SOURCES)))

all: $(EXECS)

$(BINDIR)/test: $(ALL_OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(DEBUG) -o $@ $^ -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(DEBUG) -c $< -o $@ 

$(OBJDIR)/tri.o: $(SRCDIR)/tri.cu $(OBJDIR)/point.o $(OBJDIR)/circle.o
$(OBJDIR)/delaunay.o: $(SRCDIR)/delaunay.cu $(OBJDIR)/mymath.o $(OBJDIR)/point.o $(OBJDIR)/circle.o $(OBJDIR)/tri.o

$(OBJDIR) $(BINDIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(EXECS) ./data/data.txt
