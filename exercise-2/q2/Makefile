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
CPPOUTFILE = Sobel_Demo
CPPOBJS = $(CPPSOURCE:.cpp=.o)

all: $(CPPOUTFILE)

$(CPPOUTFILE): $(CPPOBJS)
	$(CPP) $(CPPFLAGS) $(CPPOBJS) -o $(CPPOUTFILE) `pkg-config --libs opencv` $(CPPLIBS)

clean:
	rm -f *.o $(CPPOUTFILE)
