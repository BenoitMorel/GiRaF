
set(giraf-core_SOURCES 
  IO/Logger.cpp
  IO/GeneSpeciesMapping.cpp
  IO/ParallelOfstream.cpp
  maths/bitvector.cpp
  maths/Random.cpp
  trees/PLLUnrootedTree.cpp
  trees/SplitHashtable.cpp
  parallelization/ParallelContext.cpp
  )

add_library(giraf-core STATIC ${giraf-core_SOURCES})

target_include_directories(giraf-core
  PUBLIC ${GIRAF_INCLUDE_DIRS}
  )


