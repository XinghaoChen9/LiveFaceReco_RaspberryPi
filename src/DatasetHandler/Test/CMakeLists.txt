cmake_minimum_required(VERSION 3.16)
project(FaceRecognizer)

add_executable(test_image_dataset_handler test_image_dataset_handler.cpp )
target_include_directories(test_image_dataset_handler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(test_image_dataset_handler dlib::dlib)


add_executable(test_descriptor_dataset_handler test_descriptor_dataset_handler.cpp )
target_include_directories(test_descriptor_dataset_handler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(test_descriptor_dataset_handler dlib::dlib)
