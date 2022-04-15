CMAKE_MINIMUM_REQUIRED(VERSION 3.6)

PROJECT(LiveFaceReco)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ldl -lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++fs")

LINK_DIRECTORIES(${HOME_TVM}/build)

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
						
			src/livefacereco.hpp
			src/livefacereco.cpp
			src/live.cpp
			src/live.h
			src/arcface.cpp
			src/arcface.h
			src/mtcnn_new.cpp
			src/mtcnn_new.h
			src/DatasetHandler/image_dataset_handler.hpp

)

add_subdirectory(src)

if(${BUILD_TESTS})
	include(${CMAKE_MODULE_PATH}/Tests.cmake)
endif()
LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include/libs/RaspberryPi4B)
ADD_EXECUTABLE(LiveFaceReco src/main.cpp)

TARGET_LINK_LIBRARIES(LiveFaceReco ${OpenCV_LIBS}  facedetect ncnn ParallelVideoCapture pthread -lstdc++fs)
TARGET_LINK_LIBRARIES(facedetect ${OpenCV_LIBS} ncnn)
TARGET_INCLUDE_DIRECTORIES(LiveFaceReco PRIVATE ${OpenCV_INCLUDE_DIRS})
