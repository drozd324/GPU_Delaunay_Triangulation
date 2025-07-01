NVCC = nvcc
NVCCFLAGS = -Wno-deprecated-gpu-targets

EXECS = atomic_add

all: $(EXECS)

atomic_add: atomic_add.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean: 
	rm -f $(EXECS) 
