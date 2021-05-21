EXENAME = sgemv

CC      = g++
CFLAGS  = -O3 -fopenmp

CUSRCS  = $(wildcard *.cu)
OBJS    = $(CUSRCS:.cu=.o)

CUDA_PATH  = /usr/local/cuda-11.0
NVCC       = $(CUDA_PATH)/bin/nvcc
NVFLAGS    = -O3 -std=c++11 -Xcompiler -fopenmp -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -Wno-deprecated-gpu-targets
LDFLAGS    = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -Wno-deprecated-gpu-targets 

build : $(EXENAME)

$(EXENAME): $(OBJS) 
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $(EXENAME) $(OBJS)

%.o : %.cu  
	$(NVCC) $(NVFLAGS)  -c $^ 

clean:
	$(RM) *.o $(EXENAME)
