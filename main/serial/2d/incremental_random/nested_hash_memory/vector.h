#ifndef VECTOR_H
#define VECTOR_H

template <typename T>
struct vector {
	int n;
	T* array;

	vector() : n(0) {};
	vector(int num_elements) : n(num_elements), array(new T[num_elements]) {}; 
	~vector(){
		delete[] array;
	}

	T& operator[](int i) {
		return array[i];
	}

	int size() {
		return n;
	}
};

#endif
