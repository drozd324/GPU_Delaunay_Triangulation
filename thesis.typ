#import "@preview/lovelace:0.3.0": *
#import "@preview/ctheorems:1.1.3": *
#show: thmrules.with(qed-symbol: $square$)

//#set page(width: 16cm, height: auto, margin: 1.5cm)
//#set heading(numbering: "1.1.")
//
#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em))
#let theorem = thmplain( "theorem", "Theorem", titlefmt: strong)
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


#align(center)[
	#set par(justify: false)
	*Abstract*\
	An exploration of the synthesis and implementation of Delaunay triangulation algorithms
	for their use in hetrogenous computing.
]


//#figure(
//  image("gaelic_font.svg.png", width: 40%),
//  caption: [I like this font],
//)

#pagebreak()

#set heading(numbering: "1.")
#outline()

#pagebreak()

= Motivation
= Preliminary

= Serial Algorithms
== Lawsons algorithm

#theorem[
	Given any two triangulations of a set of points S, T' and T'', there exist
	a finite sequence of exchanges by which T' can be transformed to T''.
]


=== implementation
== Incremental Point Insertion 
=== implementation

= The CUDA programming model
= Parallel Algorithms
== GPU-DT
== gDel3d
=== implementation


#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(booktabs: true, numbered-title: [My cool algorithm])[
    + do something
    + *while* still something to do
      + do even more
      + *if* not done yet *then*
        + wait a bit
        + resume working
      + *else*
        + go home
      + *end*
    + *end*
  ]
)


text in here and funny thing to @NumericalRecipies
text in here and funny thing to @devadoss2011

#pagebreak()
#bibliography("references.bib")
