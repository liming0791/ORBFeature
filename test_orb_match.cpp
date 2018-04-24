#include <sys/time.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "ORBExtractor.h"
#include "gms_matcher.h"

timeval l_t_b, l_t_e;
#define TIME_BEGIN() gettimeofday(&l_t_b, 0);
#define TIME_END(TAG) gettimeofday(&l_t_e, 0);printf("===%s TIME:%lfs\n", TAG, (l_t_e.tv_sec - l_t_b.tv_sec) + (l_t_e.tv_usec - l_t_b.tv_usec) * 1e-6);


using namespace std;

int main(int argc, char** argv) {

  if (argc < 3) {
    printf("usage: %s img_name img_name\n", argv[0]);
    return 0;
  }

  cv::Mat img1 = cv::imread(argv[1]);
  cv::Mat img2 = cv::imread(argv[2]);

  ORBExtractor orb_extractor(8, 1.2, 20, 10);

  /// for img1
  TIME_BEGIN();
  orb_extractor.BuildImagePyramids(img1);
  TIME_END("buildImagePyramids");
  TIME_BEGIN();
  vector<cv::KeyPoint> key_points1 = orb_extractor.DetectKeyPoints(1500);
  TIME_END("detectkeypoints");
  TIME_BEGIN();
  cv::Mat descriptors1 = orb_extractor.ComputeDescriptors().clone();
  TIME_END("ComputeDescriptors");
  printf("key_points1.size():%ld\n", key_points1.size());

  /// for img2
  orb_extractor.BuildImagePyramids(img2);
  vector<cv::KeyPoint> key_points2 = orb_extractor.DetectKeyPoints(1500);
  cv::Mat descriptors2 = orb_extractor.ComputeDescriptors().clone();
  printf("key_points2.size():%ld\n", key_points2.size());

  /// match
  TIME_BEGIN();
  cv::flann::Index flannIndex(descriptors2, cv::flann::LshIndexParams(12,20,2), cvflann::FLANN_DIST_HAMMING);
  cv::Mat matchindex(descriptors1.rows, 2, CV_32SC1);
  cv::Mat matchdistance(descriptors1.rows, 2, CV_32SC1);
  flannIndex.knnSearch(descriptors1, matchindex, matchdistance, 2, cv::flann::SearchParams());
  TIME_END("KNN_search");

  /// 根据劳氏算法
  vector<cv::DMatch> all_matches;
  all_matches.reserve(200);
  int* _matchindex = (int*)matchindex.data;
  int* _matchdistance = (int*)matchdistance.data;
  for (int i = 0; i < matchdistance.rows; i++) {
    //printf("match_distance1:%d match_distance2:%d\n", _matchdistance[2*i], _matchdistance[2*i+1]);
    if (_matchdistance[2*i] > 512 || _matchdistance[2*i+1] > 512) continue;
    if (_matchdistance[2*i] < 0.618f*_matchdistance[2*i+1]) {
      cv::DMatch dmatches(i, _matchindex[2*i], _matchdistance[2*i]);
      all_matches.push_back(dmatches);
    }
  }
  printf("matches.size(): %ld\n", all_matches.size());

  /// draw match
  cv::Mat img_matches;
  cv::drawMatches( img1, key_points1, img2, key_points2, all_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  cv::imwrite( "matches_all.png", img_matches);

  /// use GMS match
  vector<bool> vbInliers;
  TIME_BEGIN();
  gms_matcher gms(key_points1, img1.size(), key_points2, img2.size(), all_matches);
  int num_inliers = gms.GetInlierMask(vbInliers, true, true);
  TIME_END("gms_match");
  printf("gms_num_inliers: %d\n", num_inliers);

  vector<cv::DMatch> gms_matches;
  gms_matches.reserve(500);
  for (int i = 0; i < (int)vbInliers.size(); ++i) {
    if (vbInliers[i])
      gms_matches.push_back(all_matches[i]);
  }

  /// draw match
  cv::drawMatches( img1, key_points1, img2, key_points2, gms_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  cv::imwrite( "matches_gms.png", img_matches);


  return 0;
}