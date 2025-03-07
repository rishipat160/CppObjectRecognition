#ifndef THRESHOLDING_HPP
#define THRESHOLDING_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

// Structure to hold region features
struct RegionFeatures {
    double percentFilled;       // Area / bounding box area
    double aspectRatio;         // Height / width ratio
    double hu1, hu2;            // First two Hu moments (rotation invariant)
    cv::Point2f center;         // Center of mass
    double orientation;         // Principal axis orientation
    cv::RotatedRect orientedBox; // Oriented bounding box
};

// Main Function
cv::Mat applyThreshold(cv::Mat &frame, int threshold_value, int morphOperation = 3, int kernelSize = 7, int distanceMetric = 0);

// Compute Region Features
RegionFeatures computeRegionFeatures(const cv::Mat& labelsMat, int regionId);

// Save Feature Vector
void saveFeatureVector(const RegionFeatures& features, const std::string& label);

// Print Feature Vector
void printFeatureVector(const RegionFeatures& features, int regionId);

// Classify Object with Confidence
std::pair<std::string, double> classifyObjectWithConfidence(const RegionFeatures& features, int distanceMetric = 0);

#endif 