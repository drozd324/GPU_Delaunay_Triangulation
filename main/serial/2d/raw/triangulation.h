#include <cstdlib>

struct Point {
    double x, y;

    Point() = default;
    Point(double inx, double iny) : x{inx}, y{iny}{};

    bool operator==(const Point & rhs);
    bool operator<(const Point & rhs);
    bool operator>(const Point & rhs);
};


// make this into a triply linked linst kind of thing
struct Triangle {
	Point pts[3];
	Triangle nbr[3];
	
	Triangle(Points* points) : pts{points}{};
}


// which edge do they share if any?
// should be constructed in the struct with indexing
//	pts with idx (0,1) should have nbr with idx 0 
//	pts with idx (1,2) should have nbr with idx 1 
//	pts with idx (2,0) should have nbr with idx 2
//  

struct Triangulation {
	int num_pts, num_tris;
	Point* pts;
	Triangle* tris;

	Trangulation(Point* points, int n) : pts{points}, num_pts{n} {};
	void init();
}




/**
 * @Brief Flips a triangle with one of its neighbours
 */
void flip(Triangle& tri){
	rand_num = rand() % 4;
	neighbour = tri.nbr[rand_num];
	opposite_pt_idx = rand_num % 4;
}


void Trianulation::init(){
	sort_points(pts, num_pts);

	for (int i=0; i<num_pts-3; ++i){
		Triangle tri({pts[i], pts[i+1], pts[i+2]});

	}
	
		
}


//triangle_splitting
//	- convex hull
//		- sort points
//			- point struct
//	- trianglute polygon ()  
//		- tri struct
//
