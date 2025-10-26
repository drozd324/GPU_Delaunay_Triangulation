= Guid to structure of each data file

== tri.txt
First line an interger noting the number of points in total involved int the triangulation
with the addition of 3 extra points needed for the construction of the supertriangle as the
last 3 points. In the lines following the first line we have all of the points printed
as floats. Following this block there is a space made by two newlines. Then comes a block of 
either histrory of triangulation or only the final Delaunay traingulation. Blocks first number
notes the number of triangles and in each newline relevant information of the "tri" struct
is printed with first 3 indexes of points, then 3 indexes of neighbouring triangles, then 
3 indexes of opposite points....

== coredata.csv
Data formated in a .csv style with given header for the purpose of copying last line into
other file used to process the data.

== flipedPerIter.txt
Conatins data about the number of triangles flipped in each iteration and pass of the algorithm.

== insertedPerIter.txt
Conatins data about the number of points inserted in each iteration of the algorithm.
