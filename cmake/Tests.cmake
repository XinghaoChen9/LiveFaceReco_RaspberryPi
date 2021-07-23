
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(TESTS_DIR ${CMAKE_SOURCE_DIR}/tests/)

##################### TEST PARALLEL VIDEO CAPTURE #####################
ADD_EXECUTABLE(gtest_parallel_video_capture ${TESTS_DIR}/ParallelVideoCapture/gtest_parallel_video_capture.cpp)

TARGET_LINK_LIBRARIES(gtest_parallel_video_capture ParallelVideoCapture ${OpenCV_LIBS} gtest_main)

TARGET_INCLUDE_DIRECTORIES(gtest_parallel_video_capture PUBLIC  ${CMAKE_SOURCE_DIR}/src/ParallelVideoCapture)
#######################################################################
