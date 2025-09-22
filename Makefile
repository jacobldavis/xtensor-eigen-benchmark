CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native
SOURCE = main.cpp
TARGET = benchmark

INCLUDES = -I/usr/include/eigen3 -I/usr/local/include

LIBS = -lm

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET) benchmark_debug benchmark_fast

run: $(TARGET)
	./$(TARGET)

.PHONY: debug fast clean test