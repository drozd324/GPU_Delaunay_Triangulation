NVCC = nvcc

simpleMultiGPU : simpleMultiGPU.cu
	$(NVCC) -o simpleMultiGPU simpleMultiGPU.cu


.PHONY: clean
clean:
	rm -f simpleMultiGPU
