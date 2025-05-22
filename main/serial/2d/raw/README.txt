This should contain a bad delaunay triangualtion algo as proposed by Lawson

so  just to keep  me on track
make a triangulation algorithm
i need to make a 

point struct with some methods

a triangulation class


INCREMENTAL Triangulation Algorithm

- Sort the points of S according to x-coordinates.

- The first three points determine a triangle. Consider the next point p in the ordered set
  and connect it with all previously considered points { p1, . . . , pk}
  which are visible to p. 

- Continue this process of adding one point
  of S at a time until all of S has been processed.


then a basic edge flipping delaunay trangulation algo

EDGE FLIPPING Delaunay Triangulation Algorithm
Let S be a point set in general position, with no four points
cocircular. Start with any triangulation T. If T has an illegal edge,
flip the edge and make it legal. Continue flipping illegal edges,
moving through the flip graph of S in any order, until no more illegal
edges remain.



Then turn this into the random incremental algorithm from Numerical recipies
