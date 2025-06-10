CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++23
DEBUG := # -g -fsanitize=address -lefence

OBJDIR := obj
BINDIR := bin
SRCDIR := src

EXEC_NAMES := main
EXECS := $(addprefix $(BINDIR)/, $(EXEC_NAMES))

CPP_SOURCES := main.cpp math.cpp point.cpp circle.cpp tri.cpp delaunay.cpp
ALL_OBJS := $(addprefix $(OBJDIR)/, $(patsubst %.cpp, %.o, $(CPP_SOURCES)))

all: $(EXECS)

$(BINDIR)/main: $(ALL_OBJS) | $(BINDIR)
	$(CXX) -o $@ $^ $(DEBUG) $(CXXFLAGS) -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJDIR)/tri.o: $(SRCDIR)/tri.cpp $(OBJDIR)/point.o $(OBJDIR)/circle.o
$(OBJDIR)/delaunay.o: $(SRCDIR)/delaunay.cpp $(OBJDIR)/math.o $(OBJDIR)/point.o $(OBJDIR)/circle.o $(OBJDIR)/tri.o

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(EXECS)
