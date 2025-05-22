#ifdef __CUDACC__
#define CUDA __host__ __device__
#else
#define CUDA
#endif 

class Foo {
public:
    CUDA Foo() {}
    CUDA ~Foo() {}
    CUDA void aMethod() {}
};


// remeber to initialize everything cuz it make it templates on compile
