# A simple Makefile for compiling small SDL projects

# Set the compiler
CXX := g++

# Compiler flags (for compiling source files)
CXXFLAGS := -O2 -std=c++17 -Wall -g
# Linker flags and libraries (for linking the executable)
LDFLAGS := $(shell sdl2-config --libs) -lm  

# Add header files here
SRCS := automata.cpp

# Generate a list of dependency files
DEPS := $(SRCS:.cpp=.d)

# Name of executable
EXEC := main

# Default target
all: $(EXEC)
asm: automata.s
	as -alhnd automata.s > automata.lst

automata.s: $(SRCS)
	$(CXX) $(CXXFLAGS) automata.cpp -o automata.s


run: $(EXEC)
	./$(EXEC)

perf: $(EXEC)
	sudo perf record --call-graph dwarf ./$(EXEC)
	sudo perf script | inferno-collapse-perf > stacks.folded
	cat stacks.folded | inferno-flamegraph > flamegraph.svg
	xdg-open flamegraph.svg

# Link the object files and create the executable
$(EXEC): $(SRCS:.cpp=.o)
	$(CXX) $(LDFLAGS) $^ -o $@

# Compile the source files and generate dependency files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# Include the generated dependency files
-include $(DEPS)

.PHONY: all clean

clean:
	rm -f $(EXEC) $(SRCS:.cpp=.o) $(DEPS) automata.s

