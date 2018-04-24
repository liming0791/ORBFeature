#include <iostream>
#include <opencv2/opencv.hpp>

#include "ORBExtractor.h"

using namespace std;

int main(int argc, char** argv) {

  if (argc < 2) {
    printf("usage: %s img_name\n", argv[0]);
    return 0;
  }

  cv::Mat img = cv::imread(argv[1]);

  ORBExtractor orb_extractor(8, 1.2, 20, 10);
  orb_extractor.BuildImagePyramids(img);

  const vector<cv::KeyPoint> &key_points = orb_extractor.DetectKeyPoints();
  orb_extractor.ComputeDescriptors();
  vector<cv::Mat> imgs(8);
  for (auto &m:imgs)
    img.copyTo(m);
  for (auto const &kpt:key_points) {
    cv::circle(imgs[kpt.octave], kpt.pt, kpt.size, cv::Scalar(0, 255, 0));
    float dx = 5 * cos(kpt.angle);
    float dy = 5 * sin(kpt.angle);
    cv::line(imgs[kpt.octave], kpt.pt, kpt.pt + cv::Point2f(dx, dy), cv::Scalar(0, 255, 0));
  }
  for (int i = 0; i < 8; ++i)
    cv::imwrite("fast_on_level_" + to_string(i) + ".png", imgs[i]);

  const vector<cv::KeyPoint> &key_points_distributed = orb_extractor.DetectKeyPoints(1000);
  orb_extractor.ComputeDescriptors();
  vector<cv::Mat> imgs_distributed(8);
  for (auto &m:imgs_distributed)
    img.copyTo(m);
  for (auto const &kpt:key_points_distributed) {
    cv::circle(imgs_distributed[kpt.octave], kpt.pt, kpt.size, cv::Scalar(0, 255, 0));
    float dx = 5 * cos(kpt.angle);
    float dy = 5 * sin(kpt.angle);
    cv::line(imgs_distributed[kpt.octave], kpt.pt, kpt.pt + cv::Point2f(dx, dy), cv::Scalar(0, 255, 0));
  }

  for (int i = 0; i < 8; ++i)
    cv::imwrite("distributed_fast_on_level_"+to_string(i)+".png", imgs_distributed[i]);

  return 0;
}