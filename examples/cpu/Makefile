CXX      = g++
CXXFLAGS = -O2 -std=c++17 -I../../include

TARGET   = cpu_timers
SRC      = cpu_timers.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
