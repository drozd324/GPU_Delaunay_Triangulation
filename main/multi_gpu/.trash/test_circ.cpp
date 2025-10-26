#include "src/circle.h"
#include <fstream>
#include <iostream>

int main(){

	Point a(1, 0);
	Point b(-1, 0);
	Point c(0, 1);
	Point d(0, 2);

	Circle cc = circumcircle(a, b, c);

	std::ofstream myFile("circ.txt");
//	myFile << cc.center.x[0] << " " <<  cc.center.x[1] << "\n";
//	myFile << a.x[0] << " " << a.x[1] << "\n";
//	myFile << b.x[0] << " " << b.x[1] << "\n";
//	myFile << c.x[0] << " " << c.x[1] << "\n";
//	myFile << d.x[0] << " " << d.x[1] << "\n";
	
	std::cout << "center: " << cc.center.x[0] << " " <<  cc.center.x[1] << "| radius: " << cc.radius << "\n";
	std::cout << a.x[0] << " " << a.x[1] << "\n";
	std::cout << b.x[0] << " " << b.x[1] << "\n";
	std::cout << c.x[0] << " " << c.x[1] << "\n";
	std::cout << d.x[0] << " " << d.x[1] << "\n";

	std::cout << "incircle: " << incircle(d, a, b, c) << "\n";

	myFile.close();

	return 0;	
}
