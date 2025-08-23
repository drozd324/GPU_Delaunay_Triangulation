#ifndef RAN_H
#define RAN_H

#include <math.h>

struct Ran {
	// Implementation of the highest quality recommended generator. The constructor is called with
	// an integer seed and creates an instance of the generator. The member functions int64, doub,
	// and int32 return the next values in the random sequence, as a variable type indicated by their
	// names. The period of the generator is 3:138  1057.
	
	unsigned long long u, v, w;
	Ran(unsigned long long j) : v(4101842887655102017LL), w(1) {
		// Constructor. Call with any integer seed (except value of v above).
		u = j ^ v; int64();
		v = u    ; int64();  
		w = v    ; int64();
	}

	unsigned long long int64();
	double doub();
	float flot();

	void disk(double& x, double& y, int p);
	void disk(float& x, float& y, int p);

	void gaussian(double &x, double &y);
	void gaussian(float &x, float &y);
};

void shuffle(int* array, int n, int seed);

#endif
