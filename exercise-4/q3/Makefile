#
# Basic OpenCV example tested on Ubuntu
#

CPPINCLUDE_DIRS =  -I. -I/usr/include/opencv2 -I/usr/include/opencv4 -I/usr/include/opencv
#CPPLIBS = -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video
CPPLIBS = -L/usr/lib
LIBS =

CPP = g++
CFLAGS = -Wall -c -I.
CPPFLAGS = -Wall -Wextra -Wshadow -Wundef -fno-common $(CPPINCLUDE_DIRS)
LFLAGS = -Wall

CPPSOURCE = skeletal.cpp captureskel.cpp

CPPBIN =

BIN = ./bin
$(shell mkdir -p $(BIN) &> /dev/null)
BUILD = ./build
$(shell mkdir -p $(BUILD) &> /dev/null)

.PRECIOUS : $(BUILD)/%.o

all: $(BIN)/skeletal $(BIN)/captureskel

$(BUILD)/%.o : %.cpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

$(BIN)/%: $(BUILD)/%.o
	$(CPP) $(CPPFLAGS) $< -o $@ `pkg-config --libs opencv4` $(CPPLIBS)

clean:
	rm -rf $(BIN) $(BUILD)
