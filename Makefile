CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native
DEFINES = -DXTENSOR_USE_XSIMD
SOURCE = main.cpp
TARGET = benchmark

INCLUDES = -I/usr/include/eigen3 -I/usr/include

LIBS = -lm

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) $(SOURCE) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: clean