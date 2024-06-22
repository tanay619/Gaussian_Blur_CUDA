CXX = nvcc
CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
TARGET_DIR = bin
TARGET = $(TARGET_DIR)/gaussian_blur
SRCS = gaussian_blur.cu

OBJS = $(SRCS:.cu=.o)

all: clean build

build: $(TARGET)

$(TARGET): $(OBJS)
	mkdir -p $(TARGET_DIR)
	$(CXX) $(OBJS) -o $(TARGET) $(CXXFLAGS)

%.o: %.cu
	$(CXX) -c $< -o $@ $(CXXFLAGS)

run:
	$(TARGET) $(ARGS)

clean:
	rm -rf $(TARGET_DIR) $(OBJS)
