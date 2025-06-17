#include "math.h"

int pow(int a, int b) {
	int out = 1;
	for (int i=0; i<b; i++) {
		out *= a;
	}

	return out;
}
