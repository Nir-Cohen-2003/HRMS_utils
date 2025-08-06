cmake_minimum_required(VERSION 3.15)
project(mces LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Find dependencies
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(RDKit REQUIRED COMPONENTS GraphMol SmilesParse)

# Add the library
add_library(mces_filter src/fast_mol_filter/cpp_filter.cpp)

# Link against the dependencies
target_link_libraries(mces_filter PRIVATE 
    RDKit::GraphMol
    RDKit::SmilesParse
    Eigen3::Eigen 
)

# Add include directories
target_include_directories(mces_filter PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/fast_mol_filter>
)

# Install the library and headers
install(TARGETS mces_filter
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
install(FILES src/fast_mol_filter/cpp_filter.hpp DESTINATION include/fast_mol_filter)
