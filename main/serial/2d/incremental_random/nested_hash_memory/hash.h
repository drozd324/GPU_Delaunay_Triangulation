#ifndef HASH_H
#define HASH_H

#include "vector.h"

struct Hashfn {
	// Another example of an object encapsulating a hash function, allowing arbitrary fixed key sizes
	// or variable-length null terminated strings. The hash function algorithm is self-contained.

	static unsigned long long hashfn_tab[256];
	unsigned long long h;
	int n; // Size of key in bytes, when fixed size.

	Hashfn(int nn) : n(nn) {
		if (n == 1)
			n = 0; // Null terminated string key signaled by n D 0 or 1.

		h = 0x544B2FBACAAF1684LL;
		for (int j=0; j<256; j++) { 
			// Length 256 lookup table is initialized with
			// values from a 64-bit Marsaglia generator
			// stepped 31 times between each.
			for (int i=0; i<31; i++) {
				h = (h >> 7) ^ h;
				h = (h << 11) ^ h;
				h = (h >> 10) ^ h;
			}
		hashfn_tab[j] = h;
		}
	}

	unsigned long long fn(const void *key) { // Function that returns hash from key.
		int j;
		char *k = (char *)key; // Cast the key pointer to char pointer.
		h = 0xBB40E64DA205B064LL;
		j = 0;
		while (n ? j++ < n : *k) {
			// Fixed length or else until null.
			h = (h * 7664345821815920749LL) ^ hashfn_tab[(unsigned char)(*k)];
			k++;
		}
		
		return h;
	}
};
unsigned long long Hashfn::hashfn_tab[256];


template<class keyT, class hfnT> 
struct Hashtable { 
	// Instantiate a hash table, with methods for maintaining a one-to-one correspondence between
	// arbitrary keys and unique integers in a specified range.
	int nhash, nmax, nn, ng;
	vector<int> htable, next, garbg;
	vector<unsigned long long> thehash;
	hfnT hash; // An instance of a hash function object.

	Hashtable(int nh, int nv);
	// Constructor. Arguments are size of hash table and max number of stored elements (keys).

	int iget(const keyT &key); // Return integer for a previously set key.
	int iset(const keyT &key); // Return unique integer for a new key.
	int ierase(const keyT &key); // Erase a key.
	int ireserve(); // Reserve an integer (with no key).
	int irelinquish(int k); // Un-reserve an integer.

};

template<class keyT, class hfnT>
Hashtable<keyT, hfnT>::Hashtable(int nh, int nv):
	// Constructor. Set nhash, the size of the hash table, and nmax, the maximum number of elements
	// (keys) that can be accommodated. Allocate arrays appropriately.
	nhash(nh),  nmax(nv), nn(0), ng(0),
	htable(nh), next(nv), garbg(nv), thehash(nv),
	hash(sizeof(keyT))
{
	for (int j=0; j<nh; j++) {
		htable[j] = -1;
	} // Signifies empty.
}


template<class keyT, class hfnT>
int Hashtable<keyT, hfnT>::iget(const keyT &key) {
	// Returns integer in 0..nmax-1 corresponding to key, or 1 if no such key was previously stored.
	int j, k;
	unsigned long long pp = hash.fn(&key); // Get 64-bit hash
	j = (int)(pp % nhash); // and map it into the hash table.
	for (k = htable[j]; k != -1; k = next[k]) { // Traverse linked list until an exact match is found.
		if (thehash[k] == pp) {
			return k;
		}
	}
	return -1; // Key was not previously stored.
}


template<class keyT, class hfnT> 
int Hashtable<keyT, hfnT>::iset(const keyT &key) {
	// Returns integer in 0..nmax-1 that will henceforth correspond to key. If key was previously set,
	// return the same integer as before.
	int j, k, kprev;
	unsigned long long pp = hash.fn(&key); // Get 64-bit hash
	j = (int)(pp % nhash); // and map it into the hash table.

	if (htable[j] == -1) { // Key not in table. Find a free integer, either new or previously erased.
		k = ng ? garbg[--ng] : nn++;
		htable[j] = k;
	} else { // Key might be in table. Traverse list.
		for (k = htable[j]; k != -1; k = next[k]) {
			if (thehash[k] == pp)
				return k; // Yes. Return previous value.
			
			kprev = k;
		}

		k = ng ? garbg[--ng] : nn++ ; // No. Get new integer.
		next[kprev] = k;
	}

	if (k >= nmax) 
		throw("storing too many values");

	thehash[k] = pp; // Store the key at the new or previous integer.
	next[k] = -1;
	return k;
}


template<class keyT, class hfnT>
int Hashtable<keyT,hfnT>::ierase(const keyT &key) {
	// erase a key, returning the integer in 0..nmax-1 erased, or 1 if the key was not previously set.
	int j, k, kprev;
	unsigned long long pp = hash.fn(&key);
	j = (int)(pp % nhash);
	
	if (htable[j] == -1) 
		return -1; // key not previously set.

	kprev = -1;
	for (k = htable[j]; k != -1; k = next[k]) {
		if (thehash[k] == pp) { //found key. splice linked list around it.
			if (kprev == -1) 
				htable[j] = next[k];
			else 
				next[kprev] = next[k];
			
			garbg[ng++] = k; // add k to garbage stack as an available integer.
			return k;
		}
		kprev = k;
	}
	return -1;
}


template<class keyT, class hfnT>
int Hashtable<keyT,hfnT>::ireserve() {
	// Reserve an integer in 0..nmax-1 so that it will not be used by set(), and return its value.
	int k = ng ? garbg[--ng] : nn++ ;
	if (k >= nmax) 
		throw("reserving too many values");

	next[k] = -2;
	return k;
}
	
template<class keyT, class hfnT>
int Hashtable<keyT,hfnT>::irelinquish(int k) {
	// Return to the pool an index previously reserved by reserve(), and return it, or return 1 if it
	// was not previously reserved.
	if (next[k] != -2)
		return -1;

	garbg[ng++] = k;
	return k;
}


template<class keyT, class elT, class hfnT>
struct Hash : Hashtable<keyT, hfnT> {
	// Extend the Hashtable class with storage for elements of type elT, and provide methods for
	// storing, retrieving. and erasing elements. key is passed by address in all methods.

	using Hashtable<keyT,hfnT>::iget;
	using Hashtable<keyT,hfnT>::iset;
	using Hashtable<keyT,hfnT>::ierase;
	vector<elT> els;
	
	Hash(int nh, int nm) : Hashtable<keyT, hfnT>(nh, nm), els(nm) {}
	// Same constructor syntax as Hashtable.

	void set(const keyT &key, const elT &el) {
	// Store an element el.
		els[iset(key)] = el;
	}

	int get(const keyT &key, elT &el) {
	// Retrieve an element into el. Returns 0 if no element is stored under key, or 1 for success.
		int ll = iget(key);
		if (ll < 0) 
			return 0;

		el = els[ll];
		return 1;
	}

	elT& operator[] (const keyT &key) {
	// Store or retrieve an element using subscript notation for its key. Returns a reference that
	// can be used as an l-value.
		int ll = iget(key);
		if (ll < 0) {
			ll = iset(key);
			els[ll] = elT();
		}
		return els[ll];
	}

	int count(const keyT &key) {
	// Return the number of elements stored under key, that is, either 0 or 1.
		int ll = iget(key);
		return (ll < 0 ? 0 : 1);
	}

	int erase(const keyT &key) {
	// Erase an element. Returns 1 for success, or 0 if no element is stored under key.
		return (ierase(key) < 0 ? 0 : 1);
	}
};

#endif
