cmake_minimum_required(VERSION 3.10)

project(H5hut-io)

# Set exec path
set(LIBRARY_OUTPUT_PATH lib)

# Find libraries
find_package(HDF5 COMPONENTS C REQUIRED)
find_package(MPI REQUIRED)

if(NOT DEFINED ENV{H5HUT_DIR})
    message(FATAL_ERROR "Please set the H5HUT_DIR environment variable to the root of your H5hut installation.")
endif()

set(H5HUT_HOME $ENV{H5HUT_DIR})
set(H5HUT_INCLUDE_DIRS ${H5HUT_HOME}/include)
set(H5HUT_LIBRARY_DIRS ${H5HUT_HOME}/lib)
set(H5HUT_LIBRARIES ${H5HUT_LIBRARY_DIRS}/libH5hut.a)


# include and lib directories
link_directories(${H5HUT_LIBRARY_DIRS})


# Header file list
file(GLOB inc_files include/*.h)
file(GLOB src_files src/*.cpp)


# Compilation options
add_library(H5hut-io STATIC ${src_files})
set_target_properties(H5hut-io PROPERTIES POSITION_INDEPENDENT_CODE ON)


target_include_directories(H5hut-io PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${H5HUT_INCLUDE_DIRS}
)


# Link external libraries
target_link_libraries(H5hut-io
                        ${H5HUTIO_LIBRARIES}
                        ${H5HUT_LIBRARIES}
                        ${HDF5_LIBRARIES}
)