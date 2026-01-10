cmake_minimum_required(VERSION 3.16)

# Sets project name
project(thermosolver LANGUAGES CXX C)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Sets C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Finds required packages
find_package(OpenMP REQUIRED)
find_package(HIP REQUIRED)

# Allows you to include files from within those directories, without prefixing their filepaths
include_directories(src)

# Specifies .cpp and .h files to be passed to the compiler
add_executable(main.exe 
    src/main.cpp

    src/solver/solver.cpp 
    src/solver/grid.cpp
    
    src/utils/utils.cpp
    
    src/gpu/gpu_func.cpp
)

target_link_libraries(main.exe PRIVATE
    OpenMP::OpenMP_CXX
    hip::host
    hip::device
)
