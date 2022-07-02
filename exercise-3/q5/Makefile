#
# Basic OpenCV example tested on Ubuntu
#

CPPINCLUDE_DIRS =  -I. -I/usr/include/opencv2 -I/usr/include/opencv4 -I/usr/include/opencv
CPPLIBS = -L/usr/lib/
LIBS =

CPP = g++
CFLAGS = -Wall -c -I.
CPPFLAGS = -Wall $(CPPINCLUDE_DIRS)
LFLAGS = -Wall

CPPSOURCE = grayscale.cpp

CPPBIN =

BIN = ./bin
$(shell mkdir -p $(BIN) &> /dev/null)
BUILD = ./build
$(shell mkdir -p $(BUILD) &> /dev/null)

$(shell mkdir -p output/ &> /dev/null)

.PRECIOUS : $(BUILD)/%.o

all: $(BIN)/grayscale

$(BUILD)/%.o : %.cpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

$(BIN)/%: $(BUILD)/%.o
	$(CPP) $(CPPFLAGS) $< -o $@ `pkg-config --libs opencv4` $(CPPLIBS)

clean:
	rm -rf $(BIN) $(BUILD)
	rm -f output/*