CXX := g++
CXXFLAGS := -Og -Wall -Wextra -std=c++20
DEBUG := -g -fsanitize=address,undefined   #-lefence

NVCC := nvcc
NVCCFLAGS := -Wno-deprecated-gpu-targets -rdc=true -std=c++20 -O4 --use_fast_math --compiler-options -funroll-loops -arch=sm_75   -lcudadevrt
DEBUGNV = #-g -G #-Xcompiler -Wall #--ptxas-options=-v

OBJDIR := obj
BINDIR := bin
SRCDIR := src

EXEC_NAMES := test
EXECS := $(addprefix $(BINDIR)/, $(EXEC_NAMES))

CU_SOURCES := test.cu mymath.cu point.cu circle.cu tri.cu delaunay.cu ran.cu atomic.cu misc.cu
ALL_OBJS := $(addprefix $(OBJDIR)/, $(patsubst %.cu, %.o, $(CU_SOURCES)))

all: $(EXECS)

$(BINDIR)/test: $(ALL_OBJS)   | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(DEBUGNV) -o $@ $^ -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(DEBUGNV) -c $< -o $@ -lm

$(OBJDIR)/tri.o: $(SRCDIR)/tri.cu $(OBJDIR)/mymath.o $(OBJDIR)/point.o $(OBJDIR)/circle.o
$(OBJDIR)/delaunay.o: $(SRCDIR)/delaunay.cu $(OBJDIR)/mymath.o $(OBJDIR)/point.o $(OBJDIR)/circle.o $(OBJDIR)/tri.o $(OBJDIR)/atomic.o $(OBJDIR)/misc.o 

$(OBJDIR) $(BINDIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(EXECS) ./data/*
