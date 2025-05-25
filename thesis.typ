#import "@preview/algorithmic:1.0.0"
#import algorithmic: algorithm

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

// TODO
// I would like a good mathematical foundation of references, starting with lawsons papers

#pagebreak()

#set heading(numbering: "1.")
#outline()

#pagebreak()

= Motivation
= Preliminary
==

= Serial Algorithms
== Lawsons algorithm
== Incremental Point Insertion 


#algorithm({
  import algorithmic: *
  Procedure(
    "Binary-Search",
    ("A", "n", "v"),
    {
      //Comment[Initialize the search range]
      Assign[$l$][$1$]
      Assign[$r$][$n$]
      LineBreak
      While(
        $l <= r$,
        {
          Assign([mid], FnInline[floor][$(l + r) / 2$])
          IfElseChain(
            $A ["mid"] < v$,
            {
              Assign[$l$][$m + 1$]
            },
            [$A ["mid"] > v$],
            {
              Assign[$r$][$m - 1$]
            },
            Return[$m$],
          )
        },
      )
      Return[*null*]
    },
  )
})

