cmake_minimum_required(VERSION 3.21)
project(stereo_vision)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin 0.4 REQUIRED)
find_package(spdlog REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS})
add_executable(stereo_vision main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${EIGEN3_LIBS} ${Pangolin_LIBRARIES} spdlog::spdlog)
