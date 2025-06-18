NVCC = nvcc
NVCCFLAGS = -Wno-deprecated-gpu-targets

EXECS = vector_basic vector_function vector_class vector_struct

all: $(EXECS)

vector_basic: vector_basic.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

vector_function: vector_function.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

vector_class: vector_class.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

vector_struct: vector_struct.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean: 
	rm -f $(EXECS) 
