CMAKE_MINIMUM_REQUIRED(VERSION 3.6)
PROJECT(LiveFaceReco)
FIND_PACKAGE(OpenCV 4 REQUIRED)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl -lpthread")

LINK_DIRECTORIES(${HOME_TVM}/build)

#select the corresponding lib file
#LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include/libs/ubuntu)
#LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include/libs/arm64-v8a)
#LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include/libs/armeabi-v7a)
LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include/libs/RaspberryPi4B)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
		    ${PROJECT_SOURCE_DIR}/include)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include/ncnn)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

ADD_LIBRARY(facedetect STATIC
			include/FacePreprocess.h 			
			src/livefacereco.cpp
			src/livefacereco.hpp
			src/live.cpp
			src/live.h
			src/arcface.cpp
			src/arcface.h
			src/mtcnn_new.cpp
			src/mtcnn_new.h
)


ADD_EXECUTABLE(LiveFaceReco src/main.cpp)
#TARGET_LINK_LIBRARIES(LiveFaceReco ${OpenCV_LIBS} ${NCNN_LIB} facedetect libncnn.a)
TARGET_LINK_LIBRARIES(LiveFaceReco ${OpenCV_LIBS} facedetect libncnn.a)
