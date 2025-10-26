#include "hash.h"
#include "delaunay.h"
#include "vector.h"
#include "ran.h"

int main(){
	Hash<unsigned long long, int, Nullhash> my_hash(10, 10);

	int a=1, b=2, c=3;
	unsigned long long key;

	Ranhash hashfn;

	key = hashfn.int64(a) ^ hashfn.int64(b) ^ hashfn.int64(c);
	std::cout << "key: " << key;

	int set = 0;
	my_hash.set(key, set);
	std::cout << "set: " << set << "\n";

	int got;
	my_hash.get(key, got);
	std::cout << "get: " << got <<"\n";

	return 0;	
}
