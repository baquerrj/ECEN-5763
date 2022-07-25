CXX      := -c++
CXXFLAGS := -pedantic-errors -Wall -Wextra -Werror
LDFLAGS  := -L/usr/lib -lstdc++ -lm $(shell pkg-config --libs opencv4)
BUILD    := ./build
BIN		 := ./bin
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BIN)
TARGET   := lanedetector
INCLUDE  := -I./inc $(shell pkg-config --cflags opencv4)
SRC      :=						\
   $(wildcard *.cpp)			\
   $(wildcard src/*.cpp)

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
DEPENDENCIES \
         := $(OBJECTS:.o=.d)

all: CXXFLAGS += -O2
all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

-include $(DEPENDENCIES)

.PHONY: all build clean info remote

build:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN)

clean:
	-@rm -rvf $(BUILD)
	-@rm -rvf $(BIN)

remote: $(SRC) Makefile cars.xml
	-@cd ../ && tar --exclude='laneDetector/bin' \
	--exclude='laneDetector/build' \
	--exclude='laneDetector/*.avi' \
	-zcvf laneDetector.tgz laneDetector
info:
	@echo "[*] Application dir: ${APP_DIR}     "
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"