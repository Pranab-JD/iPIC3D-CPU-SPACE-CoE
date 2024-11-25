cmake_minimum_required(VERSION 3.10)

project(iPic3D)

set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

macro(DP var)
  message(NOTICE "${var} = '${${var}}'")
endmacro()

# First, do the site configuration

# This function globs for the existing configurations.
function(get_sites OUTVAR)
  file(GLOB site_incs RELATIVE "${CMAKE_SOURCE_DIR}/cmake/sites" "${CMAKE_SOURCE_DIR}/cmake/sites/*.cmake")
  set(sites "")
  foreach(inc "${site_incs}")
    string(REPLACE ".cmake" "" site "${inc}")
    list(APPEND sites "${site}")
  endforeach()
  list(FIND sites "default" def_ind)
  list(REMOVE_ITEM sites "default")
  list(PREPEND sites "default")
  set(${OUTVAR} "${sites}" PARENT_SCOPE)
endfunction()
get_sites(KNOWN_SITES)

# Set up cache var for which we are using
set(SITE "default" CACHE STRING "A predefined site configuration to use")
# Set the options for the UI
set_property(CACHE SITE PROPERTY STRINGS "${KNOWN_SITES}")
# Include it
include("${CMAKE_SOURCE_DIR}/cmake/sites/${SITE}.cmake")

# Use CMake default variable for this so add_library will
# automatically do the right thing.
option(BUILD_SHARED_LIBS "Use shared libraries if ON, static if OFF" ON)
option(USE_CATALYST "Use Catalyst adaptor" OFF)
option(USE_BATSRUS "Use BATSRUS flag" OFF)
option(USE_HDF5 "Use HDF5 library" OFF)
option(BENCH_MARK "Print tasks time" OFF)
option(USE_OPENMP "Use OpenMP in support loops" OFF) # must delete all cmake cache if you change this option

include(GNUInstallDirs)

# Find ParaView
if(USE_CATALYST)
  find_package(ParaView 5.7 REQUIRED)

  if (NOT TARGET ParaView::PythonCatalyst)
    message(STATUS
      "${CMAKE_PROJECT_NAME} requires ParaView to be built with Catalyst and "
      "Python support enabled. Please rebuild ParaView (or point to a "
      "different build of ParaView) with PARAVIEW_ENABLE_CATALYST and "
      "PARAVIEW_ENABLE_PYTHON set to TRUE")
  else()
    add_library(iPICAdaptor ${CMAKE_SOURCE_DIR}/catalyst/Adaptor.cxx)
    if (BUILD_SHARED_LIBS)
      install(TARGETS iPICAdaptor)
    endif()

    target_link_libraries(iPICAdaptor PRIVATE stdc++fs ParaView::PythonCatalyst VTK::CommonDataModel Python3::Python)
    # target_link_libraries(iPICAdaptor PRIVATE ParaView::PythonCatalyst VTK::CommonDataModel Python3::Python)
    target_include_directories(iPICAdaptor INTERFACE ${CMAKE_SOURCE_DIR}/catalyst)
    target_compile_definitions(iPICAdaptor INTERFACE USE_CATALYST)

  endif()
endif()

# Find HDF5
if (USE_HDF5)
  if(BUILD_SHARED_LIBS)
    set(HDF5_USE_STATIC_LIBRARIES OFF)
  else()
    set(HDF5_USE_STATIC_LIBRARIES ON)
  endif()
  find_package(HDF5 COMPONENTS C HL REQUIRED)


  if (NOT DEFINED HDF5_C_TARGET)
    message(STATUS "Creating a proper target for HDF5")
    #add_library(hdf5::hdf5 INTERFACE IMPORTED)
    set_target_properties(hdf5::hdf5 PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "${HDF5_C_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES}"
      )
    set(HDF5_C_TARGET hdf5::hdf5)
  endif()

  if (NOT DEFINED HDF5_C_HL_TARGET)
    message(STATUS "Creating a proper target for HDF5 HL")
    #add_library(hdf5::hdf5_hl INTERFACE IMPORTED)
    set_target_properties(hdf5::hdf5_hl PROPERTIES
      INTERFACE_LINK_LIBRARIES "hdf5::hdf5;${HDF5_C_HL_LIBRARIES}"
      )
    set(HDF5_C_HL_TARGET hdf5::hdf5_hl)
  endif()
endif()

# Find MPI and OpenMP
find_package(MPI REQUIRED COMPONENTS CXX)
if(USE_OPENMP)
find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()


#
# Source file list
#
file(
  GLOB
  src_files
  ConfigFile/src/*.cpp
  PSKOutput3D/*.cpp
  bc/*.cpp
  communication/*.cpp
  fields/*.cpp
  grids/*.cpp
  inputoutput/*.cpp
  mathlib/*.cpp
  mpidata/*.cpp
  particles/*.cpp
  performances/*.cpp
  processtopology/*.cpp
  solvers/*.cpp
  utility/*.cpp
  main/*.cpp
  )

add_library(
  iPIC3Dlib
  ${src_files}
  )

target_include_directories(
  iPIC3Dlib
  PUBLIC include
  )

#
# Link external libraries
#
if(USE_OPENMP)
  target_link_libraries(
    iPIC3Dlib
    PUBLIC MPI::MPI_CXX
    PUBLIC OpenMP::OpenMP_CXX
    PRIVATE ${HDF5_C_HL_TARGET}
    )
else()
  target_link_libraries(
    iPIC3Dlib
    PUBLIC MPI::MPI_CXX
    PRIVATE ${HDF5_C_HL_TARGET}
    )
endif()

if (USE_CATALYST)
  target_link_libraries(
    iPIC3Dlib
    PUBLIC iPICAdaptor
    PRIVATE stdc++fs
    )
endif()

#
# Macro definitions
#
if(USE_BATSRUS)
  target_compile_definitions(iPIC3Dlib PUBLIC BATSRUS)
  message(" WARNING: BATSRUS flag is active.")
else()
  message(" INFO: BATSRUS is not active.")
endif()

if(NOT USE_HDF5)
  target_compile_definitions(iPIC3Dlib PUBLIC NO_HDF5)
endif()

if (BUILD_SHARED_LIBS)
  install(TARGETS iPIC3Dlib)
endif()

#
# Executable declaration
#

# Particle solver
add_executable(
  iPIC3D
  iPIC3D.cpp
  )

target_link_libraries(
  iPIC3D
  iPIC3Dlib
  stdc++fs
  mericmpi
  #msr
  #freeipmi
  )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG_MODE)
endif()

if(BENCH_MARK)
target_compile_definitions(iPIC3Dlib PUBLIC LOG_TASKS_TOTAL_TIME)
target_compile_definitions(iPIC3D PUBLIC LOG_TASKS_TOTAL_TIME)
endif()

## to save the executable in the folder where the CMakeLists.txt file is, i.e. CMAKE_CURRENT_SOURCE_DIR
set_target_properties(iPIC3D PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

## debug releases have a _d appended to the executable
set_target_properties(iPIC3D PROPERTIES DEBUG_POSTFIX "_d")

# Here we do the RPATH stuff so the exe is relocatable.
# Make it a function to avoid variables leaking.
function(add_install_libdir_to_rpath tgt)
  if(BUILD_SHARED_LIBS)
    get_target_property(install_rpaths ${tgt} INSTALL_RPATH)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND install_rpaths "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      list(APPEND install_rpaths "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    list(REMOVE_DUPLICATES install_rpaths)
    set_target_properties(${tgt}
      PROPERTIES INSTALL_RPATH "${install_rpaths}"
      )
  endif()
endfunction()
add_install_libdir_to_rpath(iPIC3D)

install(TARGETS iPIC3D)

add_subdirectory(inputfiles)

message("Which system am I compiling for:")
message("MYHOSTNAME is ${myhostname}")
message("CMAKE_SYSTEM_PROCESSOR is ${CMAKE_SYSTEM_PROCESSOR}")
message("Compiler & compiler flags:")
message("CMAKE_CXX_COMPILER is ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")
message("HDF5_INCLUDE_DIRS is ${HDF5_INCLUDE_DIRS}")
message("HDF5_LIBRARIES   is ${HDF5_LIBRARIES}")
message("HDF5_HL_LIBRARIES is ${HDF5_HL_LIBRARIES}")
message("MPI_LIBRARIES is ${MPI_LIBRARIES}")
