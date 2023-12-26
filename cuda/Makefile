NVCC = nvcc
GCC = g++
NVCC_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3 --expt-relaxed-constexpr
GCC_FLAGS = -c -O3
LINK_FLAGS = -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 

OBJS = conv.o theta_cal.o grad_cal.o non_maximum_sup.o main.o
TARGET = program
IMG = output.bmp

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LINK_FLAGS) -o $@ $^

conv.o: conv.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

theta_cal.o: theta_cal.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

grad_cal.o: grad_cal.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

non_maximum_sup.o: non_maximum_sup.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

main.o: main.cpp
	$(GCC) $(GCC_FLAGS) -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) $(IMG)