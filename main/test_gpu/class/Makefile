NVCC=nvcc

EXECS=vector_basic vector_function vector_class vector_struct

all: $(EXECS)

vector_basic: vector_basic.cu
	$(NVCC) -o $@ $^

vector_function: vector_function.cu
	$(NVCC) -o $@ $^

vector_class: vector_class.cu
	$(NVCC) -o $@ $^

vector_struct: vector_struct.cu
	$(NVCC) -o $@ $^

clean: 
	rm -f $(EXECS) 
