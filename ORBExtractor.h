//
// Created by liming on 4/21/18.
//

#ifndef ORBEXTRACTOR_ORBEXTRACTOR_H
#define ORBEXTRACTOR_ORBEXTRACTOR_H

#include <iostream>
#include <vector>
#include <list>

#include <opencv2/opencv.hpp>

using namespace std;

class OCTTreeNode {
public:
  float left_;
  float right_;
  float top_;
  float bottom_;
  int   best_response_;
  bool  is_measured_;
  const cv::KeyPoint* best_key_point_ptr_;
  vector<const cv::KeyPoint*> key_point_ptrs_;
  OCTTreeNode(): left_(0), right_(0), top_(0), bottom_(0), best_response_(0), best_key_point_ptr_(NULL), is_measured_(false) {};
  bool Distribute(OCTTreeNode& n1, OCTTreeNode& n2, OCTTreeNode& n3, OCTTreeNode& n4);

};

class ORBExtractor {
public:
  ORBExtractor() = default;
  ORBExtractor(int level_num = 8, float scale_factor = 1.2f, int fast_threshold = 20, int min_fast_threshold = 10, int key_points_num = -1, bool length_512 = true);

  void operator()(const cv::Mat & img, vector<cv::KeyPoint> & key_points, cv::Mat & descriptors);

  void BuildImagePyramids(const cv::Mat & img);
  const vector<cv::KeyPoint> & DetectKeyPoints(int key_points_num = -1);
  const cv::Mat & ComputeDescriptors(bool length_512 = true);

private:
  int   level_num_;
  float scale_factor_;
  int   fast_threshold_, min_fast_threshold_;
  int   key_points_num_;
  bool  length_512_;

  vector<float> inv_scales_, scales_;
  vector<cv::Mat> img_pyramids_;
  vector<cv::KeyPoint> key_points_;
  cv::Mat descriptors_;

  const double RADIUS;
  const int resolution_i;
  const int resolution_j;

  static int orb_pairs_[512*4];
  static int table_x_[1200][30];
  static int table_y_[1200][30];

  void DistributeKeyPoints(const vector<cv::KeyPoint> & key_points, vector<const cv::KeyPoint*> & key_point_const_ptrs, int num = 1000);
  void GenerateOrbPairs();
  void ComputeDescriptor(cv::KeyPoint & kpt, cv::Mat & img, unsigned char * data, bool length_512 = true);
  float IC_Angle(const cv::Point & pt, cv::Mat & img);
  inline unsigned char GetPixel(const int * pair_ptr, const unsigned char * img_data, int shift, int img_step) {
    int i = (pair_ptr[0] + shift) % resolution_i;
    int j = pair_ptr[1];
    int x = table_x_[i][j];
    int y = table_y_[i][j];
    return *(img_data + x + img_step*y);
  };

};


#endif //ORBEXTRACTOR_ORBEXTRACTOR_H
