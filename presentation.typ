/*

Presentations will be the same length as the seminar talks, 20 minutes plus five to
ten minutes for questions.  We will be strict about the time and keeping in time is
part of the mark for your project.  In your presentation you should explain your project
, the computational and HPC challenges of it, what you did to overcome these challenges,
and the results.  The focus should be on what you did and the results.
I write this because it is easy to eat up too much time explaining the background of
the project topic.  You should also explain your interpretation of the results and what they mean.

- Kind of script -

= Explain project

	A DT is a angle maximising algorithm
	Useful in FEM, FINITE VOL, etc
	Lawson in 87 made maths leads to flipping allowing for traversing all possible triangulations
	Incircle theorem tells us when we should flip
	Most basic algorithm by Lawson - flip until no more flip to be flipped

= computational and hpc challenges
	
	The GPU
	Why programming on a GPU is interesing
	GPU designed for highly parallel task - DT can be highly parallel
	Main tasks - make parallel pt instertion algo
			   - make parallel flip algo

= what we did to overcome these challenges

	Point inertion easy no prob
	- make tri data struct
	- use the geometry to your advantage with indexing

	Flip bit needs a lot of book keeping
	- do incirlce test on all triangles edges
	- use atomic min to pick out non conflic triangles
	- flip triangles which can be flipped
	- do again

	Yay!

= results and interpretations

	Speedup only after 10^3 pts make sense but defo can improve	

	Doesnt scale too well - time and normal time (time/#cores * clock speed)
	Make better graph
	
	Double precision take 10 longer at 10^6 pts, but sometimes necessary as
	flipping tends to get stuck with high num of pts, solution make an 
	adaptive precision compute


*/

#import "@preview/slydst:0.1.4": *
#import "@preview/lovelace:0.3.0": *
#import "@preview/subpar:0.2.2"
#set text(size: 10pt)

#show: slides.with(
  title: "Delaunay Triangulations on the GPU",
  subtitle: none,
  date: "05/09/2025",
  authors: ("Patryk Drozd",),
  layout: "medium",
  ratio: 4/3,
  title-color: none,
  
)

== 
#align(center, [
	#set text(size: 15pt)
	*MSc for High-performance Computing*
	#set text(size: 10pt)

	#v(2%)
	drozdp\@tcd.ie

	#v(4%)
	#image("./images/Trinity_Main_Logo.jpg", width: 90%)

])


//= Why?

== What is a Triangulation
//#figure(image("main/plotting/triangulation_grid/tri100_0.png", height: 100%))

#definition[
	A _triangulation_ of a planar point set $P$ is a subdivision of the
	plane determined by a maximal set of non-crossing edges whose vertex
	set is $P$.
]

#v(10%)
#subpar.grid(
	figure(image("images/triangulation1.png", width: 80%), caption: []), <a>,
	figure(image("images/triangulation2.png", width: 80%), caption: []), <b>,
	figure(image("images/triangulation3.png", width: 80%), caption: []), <c>,

	columns: (1fr, 1fr, 1fr),
	caption: [Examples of two triangulations (a) (b) on the same set of points.
			  In (c) an illustration of a non maximal set of edges.
	],
	align: bottom,
)


== What is a Delaunay Triangulation
#definition[
	A _Delaunay triangulation_ of a point set $P$ is a triangulation such that the circumsphere of any
	triangle in the triangulation does not contain any other point in $P$.
] <delaunay_def>

#subpar.grid(
	figure(image("images/flip1.png", width: 90%), caption: [
	]), <a>,

	figure(image("images/flip2.png", width: 90%), caption: [
	]), <b>,

	columns: (1fr, 1fr),
	align: bottom,
	label: <triangulations>,
)


== Why Delaunay Triangulations
#figure(image("images/blobs.jpg", height: 100%))

= The Algorithm

//== Serial Algorithm (i)
//#v(10%)
//#figure(
//	kind: "algorithm",
//	supplement: [Algorithm],
//
//	pseudocode-list(booktabs: true, numbered-title: [Randomized incremental point insertion])[
//	  Data: point set $P$ \
//	  Out: Delaunay triangulation $T$
//	  + Initialize $T$ with a triangle enclosing all points in $P$ 
//	  + Compute a random permutation of $P$
//	  + *for* $p in P$
//		  + Insert $p$ into the triangle $t$ containing it
//		//+ Legalize each edge in $t$ *recursively* 
//		  + *for* each edge $e in t$
//			  + LegalizeEdge($e$, $t$)
//	  + return $T$
//	]
//) 
//
//== Serial Algorithm (ii)
//#v(20%)
//#figure(
//	kind: "algorithm",
//	supplement: [Algorithm],
//
//	pseudocode-list(booktabs: true, numbered-title: [LegalizeEdge])[
//	  Data: egde $e$, triangle $t_a$
//	  + *if* $e$ is illegal
//		  + _flip_ with triangle $t_b$ across edge $e$
//		  + let $e_1$, $e_2$ be the outward facing edges of $t_b$
//		  + LegalizeEdge($e_1$, $t_b$)
//		  + LegalizeEdge($e_2$, $t_b$)
//	]
//) <legedge_alg>



== Parallel Algorithm
#set text(size: 8pt)
#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(booktabs: true, numbered-title: [Parallel point insertion and flipping])[
	Data: A point set $P$ \
	Out: Delaunay Triangulation $T$
	+ Initialize $T$ with a triangle $t$ enclosing all points in $P$ 
	+ Initialize locations of $p in P$ to all lie in $t$
	+ *while* there are $p in P$ to insert 
		+ *for each* $p in P$ *do in parallel*
			+ choose $p_t in P$ to insert if any
		+ *for each* $t in T$ with $p_t$ to insert *do in parallel*
			+ split $t$  
		//+ *while* there are configurations to flip
		+ *while* there are illegal edges
			//+ *for each* triangle $t in T$ mark whether it should be flipped or not.
			+ *for each* triangle $t in T$ *do in parallel*
				+ mark whether it should be flipped
			+ *for each* triangle $t in T$ in a configuration marked to flip *do in parallel*
				+ flip $t$
		+ update locations of $p in P$
	+ return $T$
  ]
) <ppi_alg>

#set text(size: 10pt)


== Visualization
#v(8%)
#subpar.grid(
	figure(image("main/plotting/triangulation_history/DT_iter1.png", width: 145%)),
	figure(image("main/plotting/triangulation_history/DT_iter2.png", width: 145%)),
	figure(image("main/plotting/triangulation_history/DT_iter4.png", width: 145%)),
	figure(image("main/plotting/triangulation_history/DT_iter5.png", width: 145%)),

	figure(image("main/plotting/triangulation_history/DT_iter7.png", width: 145%)),
	figure(image("main/plotting/triangulation_history/DT_iter8.png", width: 145%)),
	figure(image("main/plotting/triangulation_history/DT_iter10.png", width: 145%)),
	figure(image("main/plotting/triangulation_history/DT_iter11.png", width: 145%)),

	rows: (auto, auto),
	columns: (auto, auto, auto, auto),
)

== Visualization
#subpar.grid(
	figure(image("main/plotting/triangulation_grid/tri100_0.png", width: 135%) ),
	figure(image("main/plotting/triangulation_grid/tri500_0.png", width: 135%) ),
	figure(image("main/plotting/triangulation_grid/tri1000_0.png", width: 135%)),

	figure(image("main/plotting/triangulation_grid/tri100_2.png", width: 135%) ),
	figure(image("main/plotting/triangulation_grid/tri500_2.png", width: 135%) ),
	figure(image("main/plotting/triangulation_grid/tri1000_2.png", width: 135%)),

	rows: (auto, auto),
	columns: (auto, auto, auto),
)

= Challenges

== GPU programming
#v(10%)
#figure(
	image("images/grid_of_thread_blocks.png"),
		caption: [An illustration of the structure of the GPU programming model.
	]
) 

== Parallel point insertion
	
#v(10%)
#subpar.grid(
	figure(image("images/s_insert1.png"), caption: [
		Before insertion.
	]), <a>,

	figure(image("images/p_insert.png"), caption: [
		After insertion.
	]), <b>,

	columns: (1fr, 1fr),
	//caption: [*UPDATE THIS*],
	align: bottom,
	label: <full>,
)

== Parallel flipping

#v(17%)
#subpar.grid(
	figure(image("images/pflip0.png"), caption: [
		Pass 0
	]), <a>,

	figure(image("images/pflip1.png"), caption: [
		Pass 1
	]), <b>,

	figure(image("images/pflip2.png"), caption: [
		Pass 2
	]), <c>,

	columns: (1fr, 1fr, 1fr),
	align: bottom,
	label: <parallel_flip_img>,
)


= Analysis

//== 
//All benchmarks performed on $10^5$ points, block size of 128 and uniform point distribuiton.

== Timing
#figure(image("main/plotting/nptsVsTime/nptsVsTime.png", width: 80%))

== Speedup
#figure(image("main/plotting/nptsVsSpeedup/nptsVsSpeedup.png", width: 80%))

//== Floating point precisions
//#figure(image("main/plotting/floatVsDouble/floatVsDouble.png", width: 80%))

== Scaling
#v(10%)
#figure(image("main/plotting/gpuModelTest/gpuModelTest.png", width: 100%))

== Block Size
#figure(image("main/plotting/blocksizeVsTime/blocksizeVsTime.png", width: 80%)) 

== Profiling
#figure( image("main/plotting/timeDistrib/timeDistrib.png", width: 72%))

== Point Insertions
#figure(image("main/plotting/ninsertVsIter/ninsertVsIter.png", width: 90%))

== Flipping
#v(15%)
#figure( image("main/plotting/nflipsVsIter/nflipsVsIter.png", width: 115%))

= Conclusion
= Thank You! Any Questions?

//Add some references
