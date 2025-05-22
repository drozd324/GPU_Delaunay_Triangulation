#include <fstream>
#include <stdlib.h>

struct Point {
    double x;
    double y;

    Point() = default;
    Point(double inx, double iny) : x{inx}, y{iny}{};

    bool operator==(const Point & rhs);
    bool operator<(const Point & rhs);
    bool operator>(const Point & rhs);
};

void sort_points(std::vector<Point>& points);

void write_to_file(std::string fn, std::vector<Point> pts);
bool cross_prod(Point p1, Point p2, Point p3);
