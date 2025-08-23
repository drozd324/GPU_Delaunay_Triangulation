CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++23
DEBUG := # -g -fsanitize=address -lefence

OBJDIR := obj
BINDIR := bin
SRCDIR := src

EXEC_NAMES := main test_circ
EXECS := $(addprefix $(BINDIR)/, $(EXEC_NAMES))
OBJS := $(addsuffix .o, $(addprefix $(OBJDIR)/, $(EXEC_NAMES)))

all: $(EXECS)

$(BINDIR)/main: main.cpp
	$(CXX) -o $@ $^ $(DEBUG) $(CXXFLAGS)

$(BINDIR)/test_circ: test_circ.cpp
	$(CXX) -o $@ $^ $(DEBUG) $(CXXFLAGS)

#$(BINDIR)/%: $(OBJDIR)/%.o | $(BINDIR)
#	$(CXX) -o $@ $^ $(DEBUG) $(CXXFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.h | $(OBJDIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJDIR) $(BINDIR):
	mkdir -p $@

# Clean target
.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(EXECS)
