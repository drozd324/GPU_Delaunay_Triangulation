#import "@preview/subpar:0.2.2"
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
	algorithms used in this project. For the entirety of this project we only focus on 2 dimensional 
	Delaunay triangulations.

	In order to introduce indroduce the Delaunay Traingulation we first must define what we 
	mean when we say triangultion. In order to create a triangualtion we need a set of points $P$ 
	which will make up the vertices of the triangles.

	#definition[
		For a point set $P$, the term edge is used to indicate any
		segment that includes precisely two points of S at its endpoints.
	] <edge_def>

	#definition[
		A _triangulation_ of a planar point set $P$ is a subdivision of the
		plane determined by a maximal set of noncrossing edges whose vertex
		set is $P$ 
		@DiscComGeom.
	] <triangulation_def>


	#subpar.grid(
		figure(image("images/triangulation1.png", width: 50%), caption: [
		]), <a>,

		figure(image("images/triangulation2.png", width: 50%), caption: [
		]), <b>,

		columns: (auto, auto),
		caption: [Examples of two traingulations on the same set of points.
		          Triangulations are not unique!],
		align: bottom,
		label: <triangulations>,
	)

	A fact about triangulations is that we know how many triangles our triangulation
	will contains only given a set of points. This will be usefull when we will be storing 
	triangles as we will allways know the number that will be created. For our purposes
	the convex hull is a boundary enclosing our triangulation will allways be known in algorithms
	in the following chapters.

	#theorem[
		@CGAlgoApp Let $P$ be a set of $n$ points in the plane, not all collinear, and let $k$
		denote the number of points in $P$ that lie on the boundary of the convex hull
		of $P$. Then any triangulation of $P$ has $2n − 2 − k$ triangles and $3n − 3 − k$ edges.
	]

	A key feature of all of the Delaunay triangulation theorems we will be considering is that 
	no three points from the set of points $P$ which will make up our triangulation will lie 
	on a shared line. This leads us to the following definition.
	
	#definition[
		A set of points $P$ is in _general position_ if no 3 points in $P$ 
		are colinear and that no 4 points are cocircular.
	] <genpos_def>
	
	From this point onwards we will allways assume that the point set $P$ from which 
	we obtain our triangulation will be in _general position_. This is neccesary for the 
	definitions and theorems we will define.

	#definition[
		Let $e$ be an edge of a triangulation $T_1$ , and let $Q$ be the
		quadrilateral in $T_1$ formed by the two triangles having $e$ as their
		common edge. If $Q$ is convex, let $T_2$ be the triangulation after flipping
		edge $e$ in $T_1$. We say $e$ is a _legal edge_ if $T_1 ≥ T_2$ and $e$ is an _illegal edge_
		if $T_1 < T_2$ 
		@DiscComGeom
	] <legaledge_def>
	

	#definition[
		For a point set $P$, a _Delaunay triangulation_ of $P$ is a triangulation that
		only has legal edges.
		@DiscComGeom
	] <delaunay_def>

//	#theorem("Delaunay")[
//		Let P be a set of points in the plane, and let T be a triangulation
//		of P. Then T is a Delaunay triangulation of P if and only if the circumcircle of
//		any triangle of T does not contain a point of P in its interior.
//	]

	#theorem("Empty Circle Property")[
		Let $P$ be a point set in general position, where no four points are cocircular.
		A triangulation $T$ is a Delaunay triangulation if and only if no point from $P$
		is in the interior of any circumcircle of a triangle of $T$. 
		@DiscComGeom
	] <emptyCirclyProp_thrm>

	@emptyCirclyProp_thrm is the key ingredient in the the Delaunay triangulation algorithms
	we are going to use. This is because instead of having to compare angles, as is defined
	by @delaunay_def we are allowed to only perform a computation, involving finding a
	circumcicle and performing one comparison which would involve determining whether the
	point not shared by triangles circumcircle is contained inside the circumcircle or not.

	#subpar.grid(
		figure(image("images/flip1.png"), caption: [
		]), <a>,

		figure(image("images/flip2.png"), caption: [
		]), <b>,

		columns: (1fr, 1fr),
		caption: [Demonstration of the flipping operation.
			In (a) A configuration that needs to be flipped illustrated by the circumcircle of 
			$t_1$ containg the auxillary point of $t_2$ in its interior.
			In (b) configuration (a) which has been flipped and no longer needs to be
			flipped as illustrated by the both circumcirles of $t_1$ and $t_2$.
		],
		align: bottom,
		label: <triangulations>,
	)

	#theorem("Lawson")[
		@Lawson72 Given any two triangulations of a set of points $P$, $T_1$ and $T_2$, there exist
		a finite sequence of exchanges (flips) by which $T_1$ can be transformed to $T_2$.
	] <lawson_thrm>



//	#algorithm[
//		@DiscComGeom Let $S$ be a point set in general position, with no four points
//		cocircular. Start with any triangulation $T$. If $T$ has an illegal edge,
//		flip the edge and make it legal. Continue flipping illegal edges,
//		moving through the flip graph of S in any order, until no more illegal
//		edges remain.
//	]

// some definitions : 
//                  : Lawsons original transforming triangulations
// some theorems: 
//              : Lawsons original transforming triangulations


//intuition for the key concepts


//= Background
// which background is importa:nt and why

= The GPU
	The Graphical Processing Unit (GPU) is a type of hardware accelerator originally used to
	significantly improve running video rendering tasks such for example in video games through 
	visualizing the two or three dimensional environments the "gamer" would be interacting with
	or rendering vidoes in movies. Many different hardware accelerators have been tried and tested
	for more general use, like Intels Xeon Phis, however the more purpose oriented GPU has prevailed
	in the market mainly lead by Nvidia and AMD, with intel now recently entering the GPU market
	with their Arc series. Today, the GPU has gained a more general purporse status with the rise
	of General Purpose GPU (GPGPU) programming as more and more people have notices that GPUs are
	very useful as a general hardware accelerator.

	The triadional CPU (based on the Von Neumann architecture) which
	is built to perform _serial_ tasks with helful features such as branch predicion, for 
	dealing with if statements and vairable lenght instructions like for loops with
	variable lengh, The CPU is built to be a general purpose hardware for performing all tasks
	a user would demand from the computer. In contrast the GPU can't run alone and must be
	used in conjuction to the CPU. The CPU sends compute intructions for the GPU to perform and 
	data is commonly passes between the CPU and GPU. 

	#figure(
		image("images/array_processor.png", width: 80%),
		caption: [The _Single Instruction Multiple Threads (SIMT)_ classification, originally known
				  as an _Array Processor_ as illustriated by Michael J. Flynn @FlynnsTaxonomy. The 
				  control unit communicates instructions to the $N$ processing element with each
				  processing element having its own memory.]
	)

	What makes the GPU increadibly usefull in certain usecases (like the one of this report) is
	its architecture which is build to enable massively parallelisable tasks. In Flynn's Taxonomy
	@FlynnsTaxonomy, the GPUs architecture is based a subcategory of the Single Instruction 
	Multiple Data (SIMD) classification known as Single Instruction Multiple Threads (SIMT) also known 
	as an Array Processor. The SIMD classification allows for many proccessing units to perform the 
	same tasks on a shared dataset with the SIMT classification additionally allowing for each processing
	unit having its own memory allowing for more diverse proccessing of data.

	#figure(
		image("images/grid_of_clusters.png", width: 80%),
		caption: [@CUDACPP]
	)

	Nvida's GPUs take the SIMT model and further develop it. There are three core abstractions which 
	allow Nvidia's GPU model to be succesfull, a hierarchy of thread groups, shared memories and 
	synchronization @CUDACPP. The threads, which represent the a theoretical proccesses which encode
	programmed instrcutions, are launched together in groups of 32 known as _warps_. This 
	is the smallest unit of instructions that is executed on the GPU. The threads are further grouped 
	into _thread blocks_ which are used as a way of organising the shared memory to be used by each thread
	in this thread block. And once more the _thread blocks_ grouped into a _grid_.
	
	
// Compute on the gpu is useful because allows highly paraellisable algorthims to benenfit
// from this archicture
// A subset of SIMD, SIMT - describe the programming model, things that need to be considered,
// thread divergece,k
//   memory usage, (shared, texture), occupancy
// compare it a bit to OMP and MPI

= Algorithms

== Serial
	

=== Point insertion


	#subpar.grid(
		figure(image("images/s_insert1.png"), caption: [
		]), <a>,

		figure(image("images/s_insert2.png"), caption: [
		]), <b>,

		columns: (1fr, 1fr),
		caption: [],
		align: bottom,
		label: <full>,
	)

=== Flipping

	#subpar.grid(
		figure(image("images/insert_flip1.png"), caption: [
		]), <a>,

		figure(image("images/insert_flip2.png"), caption: [
		]), <b>,

		columns: (1fr, 1fr),
		caption: [Illustrating the flipping operation. In figure (a), point r has just been inserted
				  and the orange edges are have been marked to be checked for flipping. Two of these
				  end  edges end up beigh flipped in (b). The edges insidewould not quaify for flipping
				  as any quadrilateral would not form a convex region.
		],
		
		align: bottom,
		label: <full>,
	)




	#figure(
	  kind: "algorithm",
	  supplement: [Algorithm],

	  pseudocode-list(booktabs: true, numbered-title: [Flipping])[
//		Let $P$ be a point set in general position, with no four points
//		cocircular. Initialize $T$ as any trianulation of $P$
//		If $T$ has an illegal edge, flip the edge and make it legal.
//		Continue flipping illegal edges, moving through the flip
//		graph of $S$ in any order, until no more illegal edges remain.

		//+ Let $P$ be a point set in general position with no four points cocircular.
		+ Initialize $T$ as any trianulation of $P$

		+ *while* $T$ has an illegal edge
			+ *for* each illegal edge $e$
				+ flip $e$

		+ *return* $T$ 
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

== Parallel

	#figure(
	  kind: "algorithm",
	  supplement: [Algorithm],

	  pseudocode-list(booktabs: true, numbered-title: [Parallel point insertion and flipping @gDel3D])[
		Data: A point set $P$
		+ Initialize $T$ with a triangle $t$ enclosing all points in $P$ 
		+ *while* there are $p in P$ to insert 
			+ *for each* $p in P$ *do in parallel*
				+ choose $p_t in P$ to insert
			+ *for each* $t in T$ with $p_t$ to insert *do in parallel*
				+ split $t$  
			+ *while* there are configurations to flip
				+ *for each* base triangle $t in T$ in a configuration marked to flip
					+ flip $t$
			+ update locations of $p in P$
		+ return $T$
	  ]
	)


=== Insertion


	#subpar.grid(
		figure(image("images/s_insert1.png"), caption: [
			An image of the andromeda galaxy.
		]), <a>,

		figure(image("images/p_insert.png"), caption: [
			A sunset illuminating the sky abov.
		]), <b>,

		columns: (1fr, 1fr),
		caption: [A figure composed of two sub figures.],
		align: bottom,
		label: <full>,
	)


=== Flipping

== Data Structures

=== Triangles
	
	The core data structure that is needed in this algorithm is one to represent a the triangulation itself.
	There are a handful of different approaches to this problem inculding representing edges by the qaud
	edge data structure @Guibas85 however we choose to represent the triangles in our triangulation by
	explicit triangle structures @Nanjappa12 which hold neccesary information about their neighbours for 
	the construction of the trianulation and for performing point insertion and flipping operations.

	```c
		struct Tri {
			int p[3]; // points
			int n[3]; // neighours
			int o[3]; // opposite points
		};
	```

	#figure(
		image("images/tri_struct.png", width: 40%),
		caption: [An illustration of the _Tri_ data structures main features. We describe the triangle $t_i$ 
				  int the figure. Oriented counter clockwise points are stored as indexes an array
				  containing two dimensional coordinate represeting the point. The neighbours are
				  assigned by using the right hand side of each edge using and index of the point
				  as the start of the edge and following the edge in the CCW direction. The neighbours 
				  index will by written in the corresponding entry in the structure. 
		]
	) <tri_stuct>



#pagebreak()
#bibliography("references.bib")
