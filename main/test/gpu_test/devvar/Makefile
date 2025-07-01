NVCC = nvcc
NVCCFLAGS = -Wno-deprecated-gpu-targets

EXECS = devvar

all: $(EXECS)

devvar: devvar.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean: 
	rm -f $(EXECS) 
