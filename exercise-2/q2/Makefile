#
# Basic OpenCV example tested on Ubuntu
#

CPPINCLUDE_DIRS =  -I. -I/usr/local/include/opencv2 -I/usr/local/include/opencv4 -I/usr/local/include/opencv
CPPLIBS = -L/usr/local/lib -lopencv_core -lopencv_flann -lopencv_video
LIBS =

CPP = g++
CFLAGS = -Wall -c -I.
CPPFLAGS = -Wall $(CPPINCLUDE_DIRS)
LFLAGS = -Wall

CPPSOURCE = Sobel_Demo.cpp

CPPBIN =

BIN = ./bin
$(shell mkdir -p $(BIN) &> /dev/null)
BUILD = ./build
$(shell mkdir -p $(BUILD) &> /dev/null)

.PRECIOUS : $(BUILD)/%.o

all: $(BIN)/Sobel_Demo

$(BUILD)/%.o : %.cpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

$(BIN)/%: $(BUILD)/%.o
	$(CPP) $(CPPFLAGS) $< -o $@ `pkg-config --libs opencv` $(CPPLIBS)

clean:
	rm -rf $(BIN) $(BUILD)
