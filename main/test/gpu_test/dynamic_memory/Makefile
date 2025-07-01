NVCC = nvcc
NVCCFLAGS = -Wno-deprecated-gpu-targets

EXECS = dynmem

all: $(EXECS)

dynmem: dynmem.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean: 
	rm -f $(EXECS) 
