cmake_minimum_required(VERSION 3.8)
project(ORBExtractor)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp ORBExtractor.cc ORBExtractor.h)
target_link_libraries(main ${OpenCV_LIBRARIES})

add_executable(test_fast test_fast.cpp ORBExtractor.cc ORBExtractor.h)
target_link_libraries(test_fast ${OpenCV_LIBRARIES})

add_executable(test_orb_match test_orb_match.cpp ORBExtractor.cc ORBExtractor.h)
target_link_libraries(test_orb_match ${OpenCV_LIBRARIES})
