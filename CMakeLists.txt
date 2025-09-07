cmake_minimum_required(VERSION 3.10)

project(iPic3D)

set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Macro for debugging output
macro(DP var)
    message(NOTICE "${var} = '${${var}}'")
endmacro()

############### SITE CONFIGURATION ###############

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

set(SITE "default" CACHE STRING "A predefined site configuration to use")
set_property(CACHE SITE PROPERTY STRINGS "${KNOWN_SITES}")
include("${CMAKE_SOURCE_DIR}/cmake/sites/${SITE}.cmake")

############### BUILD OPTIONS ###############

option(BUILD_SHARED_LIBS "Use shared libraries if ON, static if OFF" ON)
option(USE_CATALYST "Use Catalyst adaptor" OFF)
option(USE_BATSRUS "Use BATSRUS flag" OFF)
option(USE_HDF5 "Use HDF5 library" ON)
option(BENCH_MARK "Print tasks time" OFF)
option(USE_OPENMP "Use OpenMP in support loops" ON)
option(PROFILING "Measure runtimes of substages" OFF)
option(PROFILE_MOMENTS "Profile Moment Gatherer" OFF)
option(PROFILE_FIELDS "Profile Field Solver" OFF)

include(GNUInstallDirs)

############### PARAVIEW/CATALYST ###############

if(USE_CATALYST)
    find_package(ParaView 5.7 REQUIRED)
    if (NOT TARGET ParaView::PythonCatalyst)
        message(FATAL_ERROR "ParaView must be built with Catalyst and Python support.")
    endif()
    add_library(iPICAdaptor ${CMAKE_SOURCE_DIR}/catalyst/Adaptor.cxx)
    if (BUILD_SHARED_LIBS)
        install(TARGETS iPICAdaptor)
    endif()
    target_link_libraries(iPICAdaptor PRIVATE stdc++fs ParaView::PythonCatalyst VTK::CommonDataModel Python3::Python)
    target_include_directories(iPICAdaptor INTERFACE ${CMAKE_SOURCE_DIR}/catalyst)
    target_compile_definitions(iPICAdaptor INTERFACE USE_CATALYST)
endif()

############### HDF5 ###############

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

############### MPI and OpenMP ###############

find_package(MPI REQUIRED COMPONENTS CXX)
if(USE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

############### Source Files ###############

file(GLOB 
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

add_library(iPIC3Dlib ${src_files})
target_include_directories(iPIC3Dlib PUBLIC include)

############### Link Libraries ###############

target_link_libraries(iPIC3Dlib PUBLIC MPI::MPI_CXX)

if(USE_OPENMP)
    target_link_libraries(iPIC3Dlib PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_HDF5)
    target_link_libraries(iPIC3Dlib PRIVATE ${HDF5_C_HL_TARGET})
endif()

if(USE_CATALYST)
    target_link_libraries(iPIC3Dlib PUBLIC iPICAdaptor PRIVATE stdc++fs)
endif()

############### Compiler Definitions ###############

if(USE_BATSRUS)
    target_compile_definitions(iPIC3Dlib PUBLIC BATSRUS)
    message("WARNING: BATSRUS flag is active.")
else()
    message("INFO: BATSRUS is not active.")
endif()

if(NOT USE_HDF5)
    target_compile_definitions(iPIC3Dlib PUBLIC NO_HDF5)
endif()

if(PROFILING)
    add_definitions(-D__PROFILING__)
endif()

if(PROFILE_MOMENTS)
    add_definitions(-D__PROFILE_MOMENTS__)
endif()

if(PROFILE_FIELDS)
    add_definitions(-D__PROFILE_FIELDS__)
endif()

if(BENCH_MARK)
    target_compile_definitions(iPIC3Dlib PUBLIC LOG_TASKS_TOTAL_TIME)
endif()

############### Executable ###############

add_executable(iPIC3D iPIC3D.cpp)
target_link_libraries(iPIC3D iPIC3Dlib stdc++fs)

if(NOT CMAKE_BUILD_TYPE)
  	set(CMAKE_BUILD_TYPE Release)
endif()

set_target_properties(iPIC3D PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} DEBUG_POSTFIX "_d")

############### RPATH Stuff ###############

function(add_install_libdir_to_rpath tgt)
    if(BUILD_SHARED_LIBS)
        get_target_property(install_rpaths ${tgt} INSTALL_RPATH)
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            list(APPEND install_rpaths "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            list(APPEND install_rpaths "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
        endif()
        list(REMOVE_DUPLICATES install_rpaths)
        set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "${install_rpaths}")
    endif()
endfunction()
add_install_libdir_to_rpath(iPIC3D)

############### Install ###############

install(TARGETS iPIC3D)
add_subdirectory(inputfiles)

############### Debug Info ###############

message("System Information:")
message("HOSTNAME               = ${myhostname}")
message("CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
message("CMAKE_CXX_COMPILER     = ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_FLAGS        = ${CMAKE_CXX_FLAGS}")
message("HDF5_INCLUDE_DIRS      = ${HDF5_INCLUDE_DIRS}")
message("HDF5_LIBRARIES         = ${HDF5_LIBRARIES}")
message("HDF5_HL_LIBRARIES      = ${HDF5_HL_LIBRARIES}")
message("MPI_LIBRARIES          = ${MPI_LIBRARIES}")
