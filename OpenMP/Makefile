.PHONY:all

G++ = g++
TARGET1 = omp.cpp
TARGET2 = omp_visit_lock.cpp
OUTPUT1 = omp
OUTPUT2 = omp_visit_lock
OMP_FLAG = -fopenmp


all:
	$(G++) $(OMP_FLAG) $(TARGET1) -o $(OUTPUT1)
	$(G++) $(OMP_FLAG) $(TARGET2) -o $(OUTPUT2)
	