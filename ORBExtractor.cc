//
// Created by liming on 4/21/18.
//

#include <fstream>
#include "ORBExtractor.h"
#include "orb_tabel.h"

bool OCTTreeNode::Distribute(OCTTreeNode &n1, OCTTreeNode &n2, OCTTreeNode &n3, OCTTreeNode &n4) {

  if (right_ <= (left_ + 1) || bottom_ <= (top_ + 1)) {
    printf("OCTTreeNode can not be distributed, return false !\n");
    return false;
  }

  float midx = (left_ + right_)/2;
  float midy = (top_ + bottom_)/2;

  n1.left_ = n3.left_ = left_;
  n2.right_ = n4.right_ = right_;
  n1.top_ = n2.top_ = top_;
  n3.bottom_ = n4.bottom_ = bottom_;
  n1.right_ = n3.right_ = n2.left_ = n4.left_ = midx;
  n1.bottom_ = n2.bottom_ = n3.top_ = n4.top_ = midy;

  for (auto pkpt:key_point_ptrs_) {
    const cv::Point2f &pt = pkpt->pt;
    int resp = pkpt->response;
    if (pt.x < midx) {
      if (pt.y < midy) {  // n1
        n1.key_point_ptrs_.push_back(pkpt);
        if (resp > n1.best_response_) {
          n1.best_response_= resp;
          n1.best_key_point_ptr_= pkpt;
        }
      } else {    // n3
        n3.key_point_ptrs_.push_back(pkpt);
        if (resp > n3.best_response_) {
          n3.best_response_= resp;
          n3.best_key_point_ptr_= pkpt;
        }
      }
    } else {
      if (pt.y < midy) {  // n2
        n2.key_point_ptrs_.push_back(pkpt);
        if (resp > n2.best_response_) {
          n2.best_response_ = resp;
          n2.best_key_point_ptr_= pkpt;
        }
      } else {    // n4
        n4.key_point_ptrs_.push_back(pkpt);
        if (resp > n4.best_response_) {
          n4.best_response_ = resp;
          n4.best_key_point_ptr_= pkpt;
        }
      }
    }
  }

  return true;
}

ORBExtractor::ORBExtractor(int level_num, float scale_factor, int fast_threshold, int min_fast_threshold, int key_points_num, bool length_512)
    :level_num_(level_num), scale_factor_(scale_factor), fast_threshold_(fast_threshold),
     min_fast_threshold_(min_fast_threshold), key_points_num_(key_points_num), length_512_(length_512),
     RADIUS(20), resolution_i(1200), resolution_j(30){
  //GenerateOrbPairs();
  printf("level_num: %d scale_factor: %f fast_threshold: %d min_fast_threshold: %d\n", level_num_, scale_factor_, fast_threshold_, min_fast_threshold_);
}

//int ORBExtractor::orb_pairs_[512*4] = {};
//int ORBExtractor::table_x_[1500][100] = {};
//int ORBExtractor::table_y_[1500][100] = {};

void ORBExtractor::BuildImagePyramids(const cv::Mat &img) {
  inv_scales_.resize(level_num_);
  scales_.resize(level_num_);
  img_pyramids_.resize(level_num_);
  if (img.channels() == 4) {
    cv::cvtColor(img, img_pyramids_[0], CV_BGRA2GRAY);
  } else if (img.channels() == 3) {
    cv::cvtColor(img, img_pyramids_[0], CV_BGR2GRAY);
  } else {
    img.copyTo(img_pyramids_[0]);
  }
  inv_scales_[0] = scales_[0] = 1;
  for (int i = 1; i < level_num_; ++i) {
    inv_scales_[i] = inv_scales_[i-1] * scale_factor_;
    scales_[i] = 1.f / inv_scales_[i];
    cv::Size new_size(int(img.cols*scales_[i]), int(img.rows*scales_[i]));
    cv::resize(img_pyramids_[i-1], img_pyramids_[i], new_size);
    printf("level_%d -- scale:%f inv_scale:%f size:%dx%d\n", i, scales_[i], inv_scales_[i], img_pyramids_[i].cols, img_pyramids_[i].rows);
  }
}

const vector<cv::KeyPoint> &ORBExtractor::DetectKeyPoints(int key_points_num) {

  vector<cv::KeyPoint> all_fast_key_points;
  all_fast_key_points.reserve(30000);

  /**
   * detect
   */
  const int grid_size = 20;
  for (int l = 0; l < level_num_; ++l) {
    const cv::Mat & proc_img = img_pyramids_[l];
    const int proc_width = proc_img.cols;
    const int proc_height = proc_img.rows;
    const int PADDING = static_cast<int>(RADIUS);
    const int right_bound = proc_width - grid_size - 3 - PADDING;
    const int bottom_bound = proc_height - grid_size - 3 - PADDING;
    printf("level_%d---proc_img.size:%dx%d\n", l, proc_width, proc_height);
    for (int grid_x = PADDING + 3; grid_x < right_bound; grid_x += grid_size) {
      for (int grid_y = PADDING + 3; grid_y < bottom_bound; grid_y += grid_size) {
        vector<cv::KeyPoint> grid_key_points;
        cv::FAST(cv::Mat(proc_img, cv::Rect(grid_x-3, grid_y-3, grid_size+6, grid_size+6)), grid_key_points, fast_threshold_, cv::FastFeatureDetector::TYPE_9_16);
        if (grid_key_points.empty()) {
          cv::FAST(cv::Mat(proc_img, cv::Rect(grid_x-3, grid_y-3, grid_size+6, grid_size+6)), grid_key_points, min_fast_threshold_, cv::FastFeatureDetector::TYPE_9_16);
        }
        if (!grid_key_points.empty()) {
          for (auto & kpt:grid_key_points) {
            kpt.pt.x = kpt.pt.x + grid_x - 3;
            kpt.pt.y = kpt.pt.y + grid_y - 3;
            kpt.octave = l;
            kpt.size = 3 * inv_scales_[l];
            //printf("response:%f\n", kpt.response);
            //printf("orientation:%f\n", kpt.angle);
          }
          all_fast_key_points.insert(all_fast_key_points.end(), grid_key_points.begin(), grid_key_points.end());
        }
        //printf("grid (%d, %d) - grid_key_points.size():%ld\n", grid_x, grid_y, grid_key_points.size());
      }
    }
  }
  printf("all_fast_key_points.size(): %ld\n", all_fast_key_points.size());

  /**
   * resize
   */
  vector<cv::KeyPoint> resized_all_key_points = all_fast_key_points;
  for (auto & kpt:resized_all_key_points) {
    kpt.pt *= inv_scales_[kpt.octave];
  }

  /**
   * distribute
   */
  key_points_.clear();
  if (key_points_num < 0) {
    key_points_ = resized_all_key_points;
  } else {
    vector<const cv::KeyPoint*> distributed_key_point_const_ptrs;
    DistributeKeyPoints(resized_all_key_points, distributed_key_point_const_ptrs, key_points_num);
    for (auto & pkpt:distributed_key_point_const_ptrs) {
      key_points_.push_back(*pkpt);
    }
  }
  printf("key_points_.size():%ld\n", key_points_.size());

  return key_points_;
}

const cv::Mat &ORBExtractor::ComputeDescriptors(bool length_512) {

  if (length_512) {
    descriptors_ = cv::Mat((int)key_points_.size(), 64, CV_8UC1);
  } else {
    descriptors_ = cv::Mat((int)key_points_.size(), 32, CV_8UC1);
  }

  for (int i = 0, _end = (int)key_points_.size(); i < _end; ++i) {
    cv::KeyPoint & kpt = key_points_[i];
    ComputeDescriptor(kpt, img_pyramids_[kpt.octave], descriptors_.ptr(i), length_512);
  }

  return descriptors_;
}

void ORBExtractor::DistributeKeyPoints(const vector<cv::KeyPoint> &key_points,
                                       vector<const cv::KeyPoint *> &key_point_const_ptrs, int num) {
  /// initial nodes
  int width = img_pyramids_[0].cols;
  int height = img_pyramids_[0].rows;
  /// Compute how many initial nodes
  list<OCTTreeNode> OCTTreeNodes;
  vector<OCTTreeNode*> ptrInitNodes;
  if (width > height) {
    const int nIni = (int) round((float) width / (float) height);
    const float hX = (float) width / nIni;
    for (int i = 0; i < nIni; i++) {
      OCTTreeNode ni;
      ni.left_ = hX * i;
      ni.right_ = hX * (i + 1);
      ni.top_ = 0;
      ni.bottom_ = height;
      ni.key_point_ptrs_.reserve(key_points.size());
      OCTTreeNodes.push_back(ni);
      ptrInitNodes.push_back(&OCTTreeNodes.back());
    }
    /// Associate points to childs
    for(auto & kpt:key_points) {
      int nodeIdx = int(kpt.pt.x/hX);
      ptrInitNodes[nodeIdx]->key_point_ptrs_.push_back(&kpt);
      ptrInitNodes[nodeIdx]->best_key_point_ptr_= &kpt;
    }
  } else {
    const int nIni = (int) round((float) height / (float) width);
    const float hX = (float) height / nIni;
    for (int i = 0; i < nIni; i++) {
      OCTTreeNode ni;
      ni.top_ = hX * i;
      ni.bottom_ = hX * (i + 1);
      ni.left_ = 0;
      ni.right_ = width;
      ni.key_point_ptrs_.reserve(key_points.size());
      OCTTreeNodes.push_back(ni);
      ptrInitNodes.push_back(&OCTTreeNodes.back());
    }
    /// Associate points to childs
    for(auto & kpt:key_points) {
      int nodeIdx = int(kpt.pt.y/hX);
      ptrInitNodes[nodeIdx]->key_point_ptrs_.push_back(&kpt);
      ptrInitNodes[nodeIdx]->best_key_point_ptr_= &kpt;
    }
  }

  /// distribute nodes
  int numToDistribute = 0;
  for(auto & ptr:ptrInitNodes) {
    if (ptr->key_point_ptrs_.size() > 1)
      numToDistribute++;
  }
  if (numToDistribute == 0) {
    printf("No Initial node to be distributed !\n");
    return;
  }

  list<OCTTreeNode>::iterator lNode = OCTTreeNodes.begin();
  /// erease empty node
  while (lNode->key_point_ptrs_.size() <= 0) {
    lNode = OCTTreeNodes.erase(lNode);
  }
  /// move iterator to the first node which can be distributed
  while (lNode->key_point_ptrs_.size() <= 1) {
    OCTTreeNodes.push_back(*lNode);
    lNode = OCTTreeNodes.erase(lNode);
  }
  /// iterative distribute nodes
  list<OCTTreeNode>::iterator flag = lNode;
  while((int)OCTTreeNodes.size() < num && numToDistribute > 0 ) {
    if (lNode == OCTTreeNodes.end() || lNode->key_point_ptrs_.size() == 1) {
      lNode = OCTTreeNodes.begin();
      flag = lNode;
      continue;
    }
    /// distribute
    vector<OCTTreeNode> nodes(4);
    bool distribute_success = lNode->Distribute(nodes[0], nodes[1], nodes[2], nodes[3]);
    if (distribute_success) {
      for (auto &n:nodes) {
        if (n.key_point_ptrs_.size() > 1) {
          OCTTreeNodes.insert(lNode, n);
          numToDistribute++;
        } else if (n.key_point_ptrs_.size() == 1) {
          OCTTreeNodes.push_back(n);
        }
      }
      // erase this node, and increase lNode
      lNode = OCTTreeNodes.erase(lNode);
    } else {
      // increase lNode
      lNode++;
    }
    numToDistribute--;
  }

  key_point_const_ptrs.clear();
  key_point_const_ptrs.reserve(OCTTreeNodes.size());
  for (auto & node:OCTTreeNodes) {
    key_point_const_ptrs.push_back(node.best_key_point_ptr_);
  }
}

void ORBExtractor::GenerateOrbPairs() {
  /**
   * build search table
   */
  for (int i = 0; i < resolution_i; ++i) {
    double radian = CV_2PI / resolution_i * i;
    for (int j = 0; j < resolution_j; ++j) {
      double radius = RADIUS / resolution_j * j;
      table_x_[i][j] = cvRound(radius * cos(radian));
      table_y_[i][j] = cvRound(radius * sin(radian));
    }
  }

  /**
   * random select pairs
   */
  srand(time(0));
  int pair_num = 0;
  while (pair_num < 512) {
    int i1 = rand()%resolution_i, j1 = rand()%resolution_j;
    int i2 = rand()%resolution_i, j2 = rand()%resolution_j;

    int dx = table_x_[i1][j1] - table_x_[i2][j2];
    int dy = table_y_[i1][j1] - table_y_[i2][j2];
    if (sqrt(dx*dx + dy*dy) < 0.2*RADIUS) continue;

    orb_pairs_[pair_num*4 + 0] = i1;
    orb_pairs_[pair_num*4 + 1] = j1;
    orb_pairs_[pair_num*4 + 2] = i2;
    orb_pairs_[pair_num*4 + 3] = j2;

    pair_num++;
  }

  /**
   * save pattern
   */
  cv::Mat orb_pattern_mat = cv::Mat::zeros(400, 400, CV_8UC3);
  for (int i = 0; i < 512; ++i) {
    int i1 = orb_pairs_[i*4 + 0];
    int j1 = orb_pairs_[i*4 + 1];
    int i2 = orb_pairs_[i*4 + 2];
    int j2 = orb_pairs_[i*4 + 3];

    int x1 = table_x_[i1][j1];
    int y1 = table_y_[i1][j1];

    int x2 = table_x_[i2][j2];
    int y2 = table_y_[i2][j2];

    cv::line(orb_pattern_mat,cv::Point(x1*10+200, y1*10+200), cv::Point(x2*10+200, y2*10+200), cv::Scalar(rand()%255, rand()%255, rand()%255));
  }
  cv::imwrite("orb_pattern.png", orb_pattern_mat);

  ofstream orb_table_file("orb_tabel.h");
  /// write orb_pairs_
  orb_table_file << "int ORBExtractor::orb_pairs_[512*4] = {";
  for (int i = 0; i < 512*4; ++i) {
    if (i == 512*4-1)
      orb_table_file << orb_pairs_[i] << "};";
    else
      orb_table_file << orb_pairs_[i] << ",";
  }
  orb_table_file << endl;
  /// write table_x_
  orb_table_file << "int ORBExtractor::table_x_[" << resolution_i <<"]" << "["<< resolution_j <<"] = {";
  for (int i = 0; i < resolution_i; ++i) {
    orb_table_file << "{";
    for (int j = 0; j < resolution_j; ++j) {
      if (j == resolution_j-1)
        orb_table_file << table_x_[i][j];
      else
        orb_table_file << table_x_[i][j] << ",";
    }
    if (i == resolution_i-1)
      orb_table_file << "} };";
    else
      orb_table_file << "},";
  }
  orb_table_file << endl;
  /// write table_y_
  orb_table_file << "int ORBExtractor::table_y_[" << resolution_i <<"]" << "["<< resolution_j <<"] = {";
  for (int i = 0; i < resolution_i; ++i) {
    orb_table_file << "{";
    for (int j = 0; j < resolution_j; ++j) {
      if (j == resolution_j-1)
        orb_table_file << table_y_[i][j];
      else
        orb_table_file << table_y_[i][j] << ",";
    }
    if (i == resolution_i-1)
      orb_table_file << "} };";
    else
      orb_table_file << "},";
  }
  orb_table_file << endl;
  orb_table_file.close();

}

float ORBExtractor::IC_Angle(const cv::Point &pt, cv::Mat &img) {
  /**
   * compute angle from circle area
   */
  unsigned char * data_ = img.data + pt.x + img.step*pt.y;
  float sum_ux = 0, sum_uy = 0;
  //const int i_step = static_cast<int>(resolution_i * 0.01);
  //const int j_step = static_cast<int>(resolution_j * 0.1);
  //for (int i = 0; i < resolution_i; i+=i_step) {
  //  for (int j = 0; j < resolution_j; j+=j_step) {

  //    int x = table_x_[i][j];
  //    int y = table_y_[i][j];
  //    int u = *(data_ + x + img.step * y);

  //    sum_ux = sum_ux + u * x;
  //    sum_uy = sum_uy + u * y;
  //  }
  //}
  const int * pair_ptr = orb_pairs_;
  for (int cnt = 0; cnt < 512; ++cnt, pair_ptr += 2) {
    int i = pair_ptr[0];
    int j = pair_ptr[1];
    int x = table_x_[i][j];
    int y = table_y_[i][j];
    int u = *(data_ + x + img.step * y);
    sum_ux = sum_ux + u * x;
    sum_uy = sum_uy + u * y;
  }
  float radian = static_cast<float>(cv::fastAtan2(sum_uy, sum_ux)/180.f*CV_PI);
  return radian;
}

void ORBExtractor::ComputeDescriptor(cv::KeyPoint &kpt, cv::Mat &img, unsigned char * data, bool length_512) {

  cv::Point pt = kpt.pt * scales_[kpt.octave];
  kpt.angle = IC_Angle(pt, img_pyramids_[kpt.octave]);

  int shift = static_cast<int>( kpt.angle / CV_2PI * resolution_i);

  unsigned char * img_data = img.data + pt.x + pt.y*img.step;
  unsigned char * data_ptr = data;
  int * pair_ptr = orb_pairs_;
  int b_size = length_512 ? 64 : 32;
  for (int b = 0; b < b_size; ++b, pair_ptr += 32, ++data_ptr) {
    unsigned char  p0 = (GetPixel(pair_ptr+0, img_data, shift, img.step) < GetPixel(pair_ptr+2, img_data, shift, img.step));
    unsigned char  p1 = (GetPixel(pair_ptr+4, img_data, shift, img.step) < GetPixel(pair_ptr+6, img_data, shift, img.step)) << 1;
    unsigned char  p2 = (GetPixel(pair_ptr+8, img_data, shift, img.step) < GetPixel(pair_ptr+10, img_data, shift, img.step)) << 2;
    unsigned char  p3 = (GetPixel(pair_ptr+12, img_data, shift, img.step) < GetPixel(pair_ptr+14, img_data, shift, img.step)) << 3;
    unsigned char  p4 = (GetPixel(pair_ptr+16, img_data, shift, img.step) < GetPixel(pair_ptr+18, img_data, shift, img.step)) << 4;
    unsigned char  p5 = (GetPixel(pair_ptr+20, img_data, shift, img.step) < GetPixel(pair_ptr+22, img_data, shift, img.step)) << 5;
    unsigned char  p6 = (GetPixel(pair_ptr+24, img_data, shift, img.step) < GetPixel(pair_ptr+26, img_data, shift, img.step)) << 6;
    unsigned char  p7 = (GetPixel(pair_ptr+28, img_data, shift, img.step) < GetPixel(pair_ptr+30, img_data, shift, img.step)) << 7;
    //printf("%x %x %x %x %x %x %x %x \n", p0, p1, p2, p3, p4, p5, p6, p7);
    *data_ptr = p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7;
  }

}

void ORBExtractor::operator()(const cv::Mat &img, vector<cv::KeyPoint> & key_points, cv::Mat & descriptors) {
  BuildImagePyramids(img);
  key_points = DetectKeyPoints(key_points_num_);
  descriptors = ComputeDescriptors(length_512_);
}
