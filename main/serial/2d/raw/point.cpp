#include "point.h"

bool Point::operator==(const Point & rhs){
	if (x == rhs.x && y == rhs.y){
		return true;
	} else {
		return false;
	}
}

bool operator<(const Point & rhs){
	if (x == rhs.x) {
		} else if (y < rhs.y){
			return true;
		} else {
			return false;
		}
	} else if (x < rhs.x){
		return true;
	} else {
		return false;
	}
}

bool operator>(const Point & rhs){
	if (x == rhs.x) {
		} else if (y > rhs.y){
			return true;
		} else {
			return false;
		}
	} else if (x > rhs.x){
		return true;
	} else {
		return false;
	}
}



void sort_points(std::vector<Point>& points){
	auto sort_2d = [](Point p1, Point p2){
		if (p1.x == p2.x){
			if (p1.y < p2.y) {
				return true;
			} else {
				return false;
			}
		} else if (p1.x < p2.x) {
			return true;	
		} else {
			return false;
		}
	};
	
	std::sort(points.begin(), points.end(),	sort_2d);
}






int compare_points(const void* p1, const void* p2) {
    const Point* a = (const Point*)p1;
    const Point* b = (const Point*)p2;

    if (a->x == b->x) {
        return (a->y > b->y) - (a->y < b->y);  // same as comparing a->y - b->y, but safe
    } else {
        return (a->x > b->x) - (a->x < b->x);
    }
}

void sort_points(Point* points, int n) {
    qsort(points, n, sizeof(Point), compare_points);
}	



bool cross_prod(Point p1, Point p2, Point p3){
	double cross = (p2.x - p1.x)*(p3.y - p1.y) - (p3.x - p1.x)*(p2.y - p1.y);
	if (cross <= 0){
		return true;
	} else {
		return false;
	}
}

void write_to_file(std::string fn, std::vector<Point> pts){
	std::ofstream out_file(fn);
	
	for (const Point &p : pts) {
        out_file << p.x << " " << p.y << '\n';
    }

    out_file.close();
}

