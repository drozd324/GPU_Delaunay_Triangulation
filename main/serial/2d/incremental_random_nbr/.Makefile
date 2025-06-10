CXX := g++
CXXFLAGS := -lm -Wall -Wextra -std=c++23
DEBUG := # -g -fsanitize=address -lefence

OBJDIR := obj
BINDIR := bin
SRCDIR := src

EXEC_NAMES := main
EXECS := $(addprefix $(BINDIR)/, $(EXEC_NAMES))
OBJS := $(addsuffix .o, $(addprefix $(OBJDIR)/, $(EXEC_NAMES)))

all: $(EXECS)

$(BINDIR)/main: $(SRCDIR)/main.cpp $(OBJDIR)/delaunay.o
	$(CXX) -o $@ $^ $(DEBUG) $(CXXFLAGS)

$(OBJDIR)/math.o: $(SRCDIR)/math.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJDIR)/point.o: $(SRCDIR)/point.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJDIR)/circle.o: $(SRCDIR)/circle.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJDIR)/tri.o: $(SRCDIR)/tri.cpp $(OBJDIR)/point.o $(OBJDIR)/circle.o
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJDIR)/delaunay.o: $(SRCDIR)/delaunay.cpp $(OBJDIR)/math.o $(OBJDIR)/point.o $(OBJDIR)/circle.o $(OBJDIR)/tri.o
	$(CXX) -c $< -o $@ $(CXXFLAGS)


#$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
#	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJDIR) $(BINDIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(EXECS)
