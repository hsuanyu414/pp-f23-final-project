NVCC = nvcc
GCC = g++
NVCC_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3 --expt-relaxed-constexpr
GCC_FLAGS = -c -O3
LINK_FLAGS = -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 

OBJS = canny.o edge_link.o main.o
TARGET = canny
IMG = output.bmp

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LINK_FLAGS) -o $@ $^

canny.o: canny.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

edge_link.o: edge_link.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

main.o: main.cpp
	$(GCC) $(GCC_FLAGS) -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) $(IMG)