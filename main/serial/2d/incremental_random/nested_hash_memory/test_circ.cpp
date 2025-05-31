#include "src/circle.h"
#include <fstream>
#include <iostream>

int main(){

	Point a(0, 1);
	Point b(0, -1);
	Point c(1, 0);
	Point d(2, 0);

	Circle cc = circumcircle(a, b, c);

	std::ofstream myFile("circ.txt");
	myFile << cc.center.x[0] << " " <<  cc.center.x[1] << "\n";
	myFile << a.x[0] << " " << a.x[1] << "\n";
	myFile << b.x[0] << " " << b.x[1] << "\n";
	myFile << c.x[0] << " " << c.x[1] << "\n";
	myFile << d.x[0] << " " << d.x[1] << "\n";
	
	std::cout << "incircle: " << incircle(d, a, b, c) << "\n";

	myFile.close();

	return 0;	
}
