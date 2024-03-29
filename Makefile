CC=g++
NVCC=/usr/local/cuda/bin/nvcc
CXXFLAGS= -O3 -Wextra -std=c++20
CUDAFLAGS= -std=c++20 -c -arch=sm_75
LIBS= -lcudart -lcublas
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include
all: matmul.o
	$(CC) -o test main.cpp matmul.o $(LIBDIRS) $(INCDIRS) $(LIBS) $(CXXFLAGS)
matmul.o: matmul.cu
	$(NVCC) $(CUDAFLAGS) matmul.cu
clean:
	rm -rf test *.o
