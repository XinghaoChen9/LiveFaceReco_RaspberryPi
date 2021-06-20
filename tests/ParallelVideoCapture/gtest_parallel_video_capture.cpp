#include <iostream>
#include <ParallelVideoCapture/parallel_video_capture.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <gtest/gtest.h>
#include <unistd.h>
// Demonstrate some basic assertions.
TEST(ParallelVideoCapture, BasicAssertions) {

  ParallelVideoCapture cap(-1);
  EXPECT_FALSE(cap.isRunning());
  EXPECT_FALSE(cap.read());
  EXPECT_FALSE(cap.grab());
  EXPECT_EQ(cap.getIntervalMs(), 30);

  cv::Mat img;
  EXPECT_FALSE(cap.retrieve(img));
  EXPECT_TRUE(cap.open(0));
  EXPECT_FALSE(cap.read());
  EXPECT_TRUE(cap.grab());
  EXPECT_TRUE(cap.retrieve(img));
  EXPECT_FALSE(img.empty());
  EXPECT_FALSE(cap.isRunning());
  EXPECT_TRUE(cap.getFrame().empty());

  cap.startCapture(); 
  usleep(1); // make the program sleeps just for the thread start;

  EXPECT_TRUE(cap.isRunning());
  EXPECT_FALSE(cap.getFrame().empty());

  cap.stopCapture();
}

