CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native
SOURCE = main.cpp
TARGET = benchmark

INCLUDES = -I/usr/include/eigen3 -I/usr/local/include

# Default target
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) -o $(TARGET)

# Alternative builds
debug:
	$(CXX) -std=c++17 -O0 -g $(INCLUDES) $(SOURCE) -o benchmark_debug

fast:
	$(CXX) -std=c++17 -Ofast -march=native $(INCLUDES) $(SOURCE) -o benchmark_fast

# Utilities
clean:
	rm -f $(TARGET) benchmark_debug benchmark_fast

run: $(TARGET)
	./$(TARGET)

.PHONY: debug fast clean test