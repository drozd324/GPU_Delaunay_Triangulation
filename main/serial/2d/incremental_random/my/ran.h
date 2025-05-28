#ifndef RAN_H
#define RAN_H

struct Ran {
	// Implementation of the highest quality recommended generator. The constructor is called with
	// an integer seed and creates an instance of the generator. The member functions int64, doub,
	// and int32 return the next values in the random sequence, as a variable type indicated by their
	// names. The period of the generator is 3:138  1057.
	
	unsigned long long u, v, w;
	Ran(unsigned long long j) : v(4101842887655102017LL), w(1){
		// Constructor. Call with any integer seed (except value of v above).
		u = j ^ v; int64();
		v = u    ; int64();  
		w = v    ; int64();
	}

	inline unsigned long long int64() {
		// Return 64-bit random integer. See text for explanation of method.
		u = u * 2862933555777941757LL + 7046029254386353087LL;
		v ^= v >> 17;
		v ^= v << 31;
		v ^= v >> 8;
		w = 4294957665U*(w & 0xffffffff) + (w >> 32);
		unsigned long long x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
		return (x + v) ^ w;
	}

	inline double doub() {
		// Return random double-precision floating value in the range 0. to 1.
		return 5.42101086242752217e-20 * int64();
	}

	inline unsigned int int32() {
		// Return 32-bit random integer.
		return (unsigned int)int64(); 
	}
};

struct Ranhash {
	// High-quality random hash of an integer into several numeric types.

	unsigned long long u, v;
	inline unsigned long long int64(unsigned long long) {
		// Returns hash of u as a 64-bit integer.
		unsigned long long v = u * 3935559000370003845LL + 2691343689449507681LL;
		v ^= v >> 21; v ^= v << 37; v ^= v >> 4;
		v *= 4768777513237032717LL;
		v ^= v << 20; v ^= v >> 41; v ^= v << 5;
		return v;
	}

	inline unsigned int int32(unsigned long long u) {
		// Returns hash of u as a 32-bit integer.
		return (unsigned int)(int64(u) & 0xffffffff) ;
	}

	inline double doub(unsigned long long u) {
		// Returns hash of u as a double-precision floating value between 0. and 1.
		return 5.42101086242752217e-20 * int64(u);
	}
};

#endif
