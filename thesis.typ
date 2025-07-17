#import "@preview/lovelace:0.3.0": *
#import "@preview/ctheorems:1.1.3": *
#show: thmrules.with(qed-symbol: $square$)

//#set page(width: 16cm, height: auto, margin: 1.5cm)
//#set heading(numbering: "1.1.")
//
#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em))
#let theorem = thmplain( "theorem", "Theorem", titlefmt: strong)
#let algorithm = thmplain( "algorithm", "Algorithm", titlefmt: strong)
#let proof = thmproof("proof", "Proof")
#let corollary = thmplain( "corollary", "Corollary", base: "theorem", titlefmt: strong)
#let example = thmplain("example", "Example").with(numbering: none)

//==================================== DOC START =========================================

#align(center, text(17pt)[
  *Delaunay Triangulations on the GPU*
])

#align(center)[
    Patryk Drozd\
    Trinity College Dublin\
    #link("mailto:drozdp@tcd.ie")
]


#figure(
	image("main/plotting/triangulation/triangulationCover.png", width: 90%),
	//caption: [I like this font],
)

#pagebreak()

// ================================== TABLE OF CONTENTS ======================================= 

#set heading(numbering: "1.")
#outline()

#pagebreak()

// ====================================== WRITING =============================================


// Delaunay triangluations are interesing because: FEM simulation, Computational geometry, 
// i think the maths is interesting


//	An exploration of the synthesis and implementation of Delaunay triangulation algorithms
//	for their use in hetrogenous computing.
#align(center)[
	#set par(justify: false)
	*Abstract*\

	Triangulations of a set of points are a very useful mathematical construction to describe 
	properties of discretised physical systems, such as modelling terrains, cars and wind turbines
	which are a commonly uses for simulations such as compulational fluid dynamics or other physical
	properties, and even have use in video games for rendiring and visualising complex geometries. To
	paint a picure you may think of a triangulation given a set of points $P$ to be a bunch of line
	segments connecting each point in $P$ in a way such that the edges are non instersecting. A  
	particulary interesting subset of triangulations are Delaunay triangulations (DT). The Delaunay
	triangulation is a triangulation which maximises all angles in each triangle of the triangulation. 
	Mathematically this gives us an interesting optimization problem which leads to some rich
	mathematical properties, at least in 2 dimensions, and for the applied size we have a good way
	to discretize space for the case of simulations with the aid of methods such as Finite Element
	and Finite Volume methods. Delaunay triangulatinos in particular are a good candiate for these
	numerical methods as they provide us with fat triangles, as opposed to skinny triangles, which
	can serve as good elements in the Finite Element method as they tend to improve accuracy
	@MeshSmoothing. 

	There are many algorithms which compute Delaunay triangulations (cite some overview paper), however
	alot of them use the the operation of 'flipping' or originally called an 'exchange' @Lawson72. This
	is a fundamental property of moving through triangulations of a set of points to with the goal of
	optaining the optimal Delaunay triangulation. This flipping operation involves a configuration
	of two triangles sharing an edge, forming a quadrilateral with its boundary. The shared egde
	between these two triangles will be swapped or flipped from the two points at its end to the
	other two points on the quadrilateral. The original agorithm motivated by (@Lawson72) is hinted
	to be us this flipping operation to iterate through different triangulations and eventually arrive
	at the Delaunay trianglution which we desire.

	With the flippig operation being at the core of the algorithm, we can notice that is has the 
	possibility of being parallelized. This is desirable as problems which commonly use the DT 
	are run with large datasets and can benefit from the highly parallelisable nature of this 
	algorithm. If we wish to parallize this idea, and start with some initial triangulation, 
	conflicts would only occur if we chose to flip a configuration of triangles which share a
	triangle. With some care, this is an avoidable situation leads to a highly scalable algorithm.
	In our case the hardware of choice will be the GPU which is designed with the SIMT model which
	is particularly well suited for this algorithm as we are mostly performing the same operations
	in each iteration of the algorithm in parallel.

	The goad of this project is to explore the Delaunay trianglion through both serial and parallel
	algorithms with the goal of presenting a easy to understand, sufficiently complex parallel
	algorthm designed with with Nvidia's CUDA programming model for runnig software on their
	GPUs.
]



//many fields
//spanning applications in engineering, aiding in simulating complex gemetries for computational 
//fluid dyamnics or structural mechanincs, in cosmology

#pagebreak()
= Delaunay triangulations
	In this section I aim to introduce triangulations and Delaunay triangulations from a mathematical 
	perspective with the foresight to help present the motivation and inspiration for the key
	algorithms used in this project. For the entirety of this project we only focus on 2 dimensional		Delaunay triangulations.

	In order to introduce indroduce the Delaaunday Traingulation we first must define what we 
	mean when we say triangultion. In order to create a triangualtion we need a set of points $P$ 
	which will make up the vertices of the triangles.

	#definition[
		A _triangulation_ of a planar point set $P$ is a subdivision of the
		plane determined by a maximal set of noncrossing edges whose vertex
		set is $P$ @DiscComGeom.
	]


	#figure(
		image("main/plotting/triangulation/triangulationCover.png", width: 50%),
		//caption: [I like this font],
	)

	

	#definition[
		A set of points $P$ is in _general position_ if no 3 points in $P$ lie
		on a line.
	]


	Throught this chapter 

// assume all points in S are in general position


#theorem[
	Let P be a set of points in the plane, and let T be a triangulation
	of P. Then T is a Delaunay triangulation of P if and only if the circumcircle of
	any triangle of T does not contain a point of P in its interior.
]

#theorem[
	@CGAlgoApp Let P be a set of n points in the plane, not all collinear, and let k
	denote the number of points in P that lie on the boundary of the convex hull
	of P. Then any triangulation of P has 2n − 2 − k triangles and 3n − 3 − k edges.
]

#theorem[
	@Lawson72 Given any two triangulations of a set of points S, T' and T'', there exist
	a finite sequence of exchanges by which T' can be transformed to T''.
]

#algorithm[
	@DiscComGeom Let S be a point set in general position, with no four points
	cocircular. Start with any triangulation T. If T has an illegal edge,
	flip the edge and make it legal. Continue flipping illegal edges,
	moving through the flip graph of S in any order, until no more illegal
	edges remain.
]

// some definitions : 
//                  : Lawsons original transforming triangulations
// some theorems: 
//              : Lawsons original transforming triangulations


//intuition for the key concepts

= Background
// which background is important and why

= Compute on the GPU
// Compute on the gpu is useful because allows highly paraellisable algorthims to benenfit
// from this archicture
// A subset of SIMD, SIMT - describe the programming model, things that need to be considered,
// thread divergece,k
//   memory usage, (shared, texture), occupancy
// compare it a bit to OMP and MPI

= Algorithms

== Serial
== Parallel

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(booktabs: true, numbered-title: [Flipping])[
	Let S be a point set in general position, with no four points
	cocircular. 

	+ Initialize $T$ as any trianulation of $P$

	If T has an illegal edge, flip the edge and make it legal.
	Continue flipping illegal edges, moving through the flip
	graph of S in any order, until no more illegal edges remain.

	+ *while* $T$ is not _Delaunay_
		+ *for* $t$ in $T$ 
			+ *for* edge $e$ in $t$ 
				+ *if* $e$ _incicle_ 
					+ flip $e$
	+ return T
  ]
)

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(booktabs: true, numbered-title: [Thorough Flipping])[
	+ Initialize $T$ as any trianulation of $P$
	+ *while* $T$ is not _Delaunay_
		+ *for* $t$ in $T$ 
			+ *for* edge $e$ in $t$ 
				+ *if* $e$ _incicle_ 
					+ flip $e$
	+ return T
  ]
)


#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(booktabs: true, numbered-title: [Randomized incremental point insertion @CGAlgoApp])[
	+ Initialize $T$ with a triangle enclosing all points in $P$ 
	+ Compute a random permutation of $P$
	+ *for* $p in P$
		+ Insert $p$ into the triangle $t$ containing it
		+ Legalize each edge in $t$ *recursively* 
	+ return $T$
  ]
)


#pagebreak()
#bibliography("references.bib")
