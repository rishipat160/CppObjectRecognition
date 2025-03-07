#include "threshold.hpp"

/**
 * Created by: Rishi Patel
 * Project 3: Object Detection and Classification
 * 
 * I know this code is not perfect, Im still playing around with it.
 * If i spent more time I would have refactored the code better so that each individual question
 * is answered through its own cpp file. I would have split up the code in this file into muliple other files.
 * 
 * However currently i was more focused on getting the results I needed and making sure it worked.
 * The "main" function is basically the applyThreshold function. I simply kept adding to it as i was 
 * going along and never changed the name or refactored it if you are confused.abort
 * 
 * TLDR: The "main" function is the applyThreshold function.
 * 
 *
 */

// Global variable to store labels
cv::Mat g_labels;

/**
 * Applies a 5x5 Gaussian blur filter to an image using separable 1D convolutions
 * 
 * @param src The source image to blur
 * @param dst The destination image to store the blurred result
 * @return -1 if source image is empty, 0 on success
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }
    dst.create(src.size(), CV_8UC3);
    cv::Mat temp(src.size(), CV_8UC3);

    // Separable 1D kernels [1 2 4 2 1]
    const int kernel[] = {1, 2, 4, 2, 1};
    const int kernelSum = 10; 

    // Horizontal pass
    for (int i = 0; i < src.rows; i++) {
        const uchar* srcRow = src.ptr<uchar>(i);
        uchar* tempRow = temp.ptr<uchar>(i);

        for (int j = 0; j < src.cols; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply horizontal kernel
            for (int k = -2; k <= 2; k++) {
                int col = j + k;
                // Border handling
                col = std::max(0, std::min(col, src.cols - 1));
                
                const uchar* pixel = srcRow + (col * 3);
                sumB += pixel[0] * kernel[k + 2];
                sumG += pixel[1] * kernel[k + 2];
                sumR += pixel[2] * kernel[k + 2];
            }

            // Store intermediate results
            tempRow[j*3] = (uchar)(sumB / kernelSum);
            tempRow[j*3 + 1] = (uchar)(sumG / kernelSum);
            tempRow[j*3 + 2] = (uchar)(sumR / kernelSum);
        }
    }
        // Vertical pass
    for (int j = 0; j < src.cols; j++) {
        for (int i = 0; i < src.rows; i++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply vertical kernel
            for (int k = -2; k <= 2; k++) {
                int row = i + k;
                // Border handling
                row = std::max(0, std::min(row, src.rows - 1));
                
                const uchar* pixel = temp.ptr<uchar>(row) + (j * 3);
                sumB += pixel[0] * kernel[k + 2];
                sumG += pixel[1] * kernel[k + 2];
                sumR += pixel[2] * kernel[k + 2];
            }

            // Store final results
            uchar* dstPixel = dst.ptr<uchar>(i) + (j * 3);
            dstPixel[0] = (uchar)(sumB / kernelSum);
            dstPixel[1] = (uchar)(sumG / kernelSum);
            dstPixel[2] = (uchar)(sumR / kernelSum);
        }
    }

    return 0;
}

/**
 * Applies a simple thresholding operation to a grayscale image
 * 
 * @param gray The source grayscale image
 * @param threshold_value The threshold value for the thresholding operation
 * @return A binary image where pixels above the threshold are set to 0 (black), and others to 255 (white)
 */
cv::Mat ThresholdFunct(cv::Mat &gray, int threshold_value) {
    cv::Mat binary(gray.size(), CV_8UC1);
    uchar* grayPtr;
    uchar* binaryPtr;
    for (int i = 0; i < gray.rows; i++) {
        grayPtr = gray.ptr<uchar>(i);
        binaryPtr = binary.ptr<uchar>(i);
        for (int j = 0; j < gray.cols; j++) {
            binaryPtr[j] = (grayPtr[j] > threshold_value) ? 0 : 255;
        }
    }
    return binary;
}

/**
 * Adjusts the grayscale image based on the saturation of the image
 * 
 * @param gray The source grayscale image
 * @param saturation The source saturation image
 * @param adjustment_strength The strength of the adjustment (default is 0.3)
 */
void adjustGrayWithSaturation(cv::Mat &gray, cv::Mat &saturation, float adjustment_strength = 0.3f) {
    for(int i = 0; i < gray.rows; i++) {
        uchar* grayPtr = gray.ptr<uchar>(i);
        uchar* satPtr = saturation.ptr<uchar>(i);
        for(int j = 0; j < gray.cols; j++) {
            float saturation_factor = satPtr[j] / 255.0f;
            grayPtr[j] = static_cast<uchar>(grayPtr[j] * (1.0f - adjustment_strength * saturation_factor));
        }
    }
}

/**
 * Performs morphological operations on a binary image
 * 
 * @param binary The source binary image
 * @param operation The type of morphological operation to perform (0: Erosion, 1: Dilation, 2: Opening, 3: Closing)
 * @param kernelSize The size of the kernel for the morphological operation (default is 3)
 */
cv::Mat performMorphology(cv::Mat &binary, int operation, int kernelSize = 3) {
    cv::Mat result = binary.clone();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    
    switch(operation) {
        case 0: 
            cv::erode(binary, result, kernel);
            break;
        case 1: 
            cv::dilate(binary, result, kernel);
            break;
        case 2: // Opening - removes small noise
            cv::erode(binary, result, kernel);
            cv::dilate(result, result, kernel);
            break;
        case 3: // Closing - fills small holes
            cv::dilate(binary, result, kernel);
            cv::erode(result, result, kernel);
            break;
        default:
            break;
    }
    return result;
}

/**
 * Finds valid regions in a binary image based on their area
 * 
 * @param stats The statistics matrix of connected components
 * @param binary The source binary image
 * @param minSize The minimum size of a valid region
 * @param numLabels The total number of labels in the binary image
 * @return A vector of valid region labels
 */
std::vector<int> findValidRegions(const cv::Mat &stats, const cv::Mat &binary, int minSize, int numLabels) {
    std::vector<int> validLabels;
    for(int label = 1; label < numLabels; label++) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if(area < minSize) continue;
        
        validLabels.push_back(label);
    }
    return validLabels;
}

/**
 * Generates a vector of colors for the regions
 * 
 * @param numColors The number of colors to generate
 * @return A vector of colors
 */
std::vector<cv::Vec3b> generateColors(size_t numColors) {
    std::vector<cv::Vec3b> colors(numColors + 1);
    colors[0] = cv::Vec3b(0, 0, 0); 
    return colors;
}

/**
 * Matches the current regions with the previous regions
 * 
 * @param validLabels The valid labels in the current frame
 * @param centroids The centroids of the regions
 * @param prevRegions The previous regions
 * @param colors The colors of the regions
 * @param currentRegions The current regions
 */
void matchWithPreviousRegions(const std::vector<int> &validLabels, const cv::Mat &centroids,
                            const std::vector<std::pair<cv::Point2d, cv::Vec3b>> &prevRegions,
                            std::vector<cv::Vec3b> &colors,
                            std::vector<std::pair<cv::Point2d, cv::Vec3b>> &currentRegions) {

    // Increase this value to make matching more tolerant
    double maxMatchDistance = 150.0; 
    
    for(size_t i = 0; i < validLabels.size(); i++) {
        int label = validLabels[i];
        cv::Point2d centroid(centroids.at<double>(label, 0), centroids.at<double>(label, 1));
        
        double minDist = maxMatchDistance; 
        int bestMatch = -1;
        
        for(size_t j = 0; j < prevRegions.size(); j++) {
            double dist = cv::norm(centroid - prevRegions[j].first);
            if(dist < minDist) {
                minDist = dist;
                bestMatch = j;
            }
        }
        
        if(bestMatch >= 0) {
            colors[i+1] = prevRegions[bestMatch].second;
        } else {
            cv::RNG rng(cv::getTickCount() + i);
            colors[i+1] = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        }
        currentRegions.push_back(std::make_pair(centroid, colors[i+1]));
    }
}

/**
 * Generates new colors for the regions
 * 
 * @param validLabels The valid labels in the current frame
 * @param centroids The centroids of the regions
 * @param colors The colors of the regions
 * @param currentRegions The current regions
 */
void generateNewColors(const std::vector<int> &validLabels, const cv::Mat &centroids,
                      std::vector<cv::Vec3b> &colors,
                      std::vector<std::pair<cv::Point2d, cv::Vec3b>> &currentRegions) {
    cv::RNG rng(cv::getTickCount());
    for(size_t i = 0; i < validLabels.size(); i++) {
        int label = validLabels[i];
        cv::Point2d centroid(centroids.at<double>(label, 0), centroids.at<double>(label, 1));
        
        colors[i+1] = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        currentRegions.push_back(std::make_pair(centroid, colors[i+1]));
    }
}

/**
 * Assigns colors to the regions
 * 
 * @param validLabels The valid labels in the current frame
 * @param centroids The centroids of the regions
 * @param prevRegionsPtr The previous regions
 * @param colors The colors of the regions
 * @param currentRegions The current regions
 */
void assignRegionColors(const std::vector<int> &validLabels, const cv::Mat &centroids,
                       std::vector<std::pair<cv::Point2d, cv::Vec3b>>* prevRegionsPtr,
                       std::vector<cv::Vec3b> &colors,
                       std::vector<std::pair<cv::Point2d, cv::Vec3b>> &currentRegions) {
    if(prevRegionsPtr && !prevRegionsPtr->empty()) {
        matchWithPreviousRegions(validLabels, centroids, *prevRegionsPtr, colors, currentRegions);
    } else {
        generateNewColors(validLabels, centroids, colors, currentRegions);
    }
}

/**
 * Colors the regions in the output image
 * 
 * @param labels The labels of the regions
 * @param validLabels The valid labels in the current frame
 * @param colors The colors of the regions
 * @param output The output image
 */

void colorRegions(const cv::Mat &labels, const std::vector<int> &validLabels,
                 const std::vector<cv::Vec3b> &colors, cv::Mat &output) {
    std::map<int, int> labelMap;
    for(size_t i = 0; i < validLabels.size(); i++) {
        labelMap[validLabels[i]] = i + 1;
    }
    
    for(int y = 0; y < output.rows; y++) {
        for(int x = 0; x < output.cols; x++) {
            int oldLabel = labels.at<int>(y, x);
            if(oldLabel > 0 && labelMap.find(oldLabel) != labelMap.end()) {
                int newLabel = labelMap[oldLabel];
                output.at<cv::Vec3b>(y, x) = colors[newLabel];
            }
        }
    }
}

/**
 * Finds connected components in a binary image
 * 
 * @param binary The source binary image
 * @param minSize The minimum size of a valid region
 * @param prevRegionsPtr The previous regions
 * @return The output image with the regions colored
 */
cv::Mat findConnectedComponents(cv::Mat &binary, int minSize = 500,
                               std::vector<std::pair<cv::Point2d, cv::Vec3b>>* prevRegionsPtr = nullptr) {
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
    
    std::vector<int> validLabels = findValidRegions(stats, binary, minSize, numLabels);
    
    cv::Mat output = cv::Mat::zeros(binary.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors = generateColors(validLabels.size());
    
    std::vector<std::pair<cv::Point2d, cv::Vec3b>> currentRegions;
    
    assignRegionColors(validLabels, centroids, prevRegionsPtr, colors, currentRegions);
    
    if(prevRegionsPtr) {
        *prevRegionsPtr = currentRegions;
    }
    
    colorRegions(labels, validLabels, colors, output);
    
    g_labels = labels.clone();
    
    return output;
}

/**
 * Computes region features from a labeled image
 * 
 * @param labelsMat The labeled image
 * @param regionId The ID of the region to compute features for
 * @return A struct containing the computed region features
 */
RegionFeatures computeRegionFeatures(const cv::Mat& labelsMat, int regionId) {
    cv::Mat regionMask = (labelsMat == regionId);
    
    cv::Moments m = cv::moments(regionMask, true);
    
    RegionFeatures features;
    
    features.center = cv::Point2f(m.m10/m.m00, m.m01/m.m00);
    
    features.orientation = 0.5 * atan2(2*m.mu11, m.mu20 - m.mu02);
    
    double theta = features.orientation;
    
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    
    double minAlongAxis = DBL_MAX, maxAlongAxis = -DBL_MAX;
    double minPerpAxis = DBL_MAX, maxPerpAxis = -DBL_MAX;
    
    for(int y = 0; y < labelsMat.rows; y++) {
        for(int x = 0; x < labelsMat.cols; x++) {
            if(labelsMat.at<int>(y, x) == regionId) {
                double alongAxis = (x - features.center.x) * cosTheta + 
                                  (y - features.center.y) * sinTheta;
                double perpAxis = -(x - features.center.x) * sinTheta + 
                                  (y - features.center.y) * cosTheta;
                
                minAlongAxis = std::min(minAlongAxis, alongAxis);
                maxAlongAxis = std::max(maxAlongAxis, alongAxis);
                minPerpAxis = std::min(minPerpAxis, perpAxis);
                maxPerpAxis = std::max(maxPerpAxis, perpAxis);
            }
        }
    }
    
    double width = maxAlongAxis - minAlongAxis;
    double height = maxPerpAxis - minPerpAxis;
    
    features.orientedBox = cv::RotatedRect(
        features.center, 
        cv::Size2f(width, height), 
        theta * 180.0 / CV_PI);
    
    features.percentFilled = m.m00 / (width * height);
    
    features.aspectRatio = width > height ? width / height : height / width;
    
    double huMoments[7];
    cv::HuMoments(m, huMoments);
    features.hu1 = -std::log10(std::abs(huMoments[0]));
    features.hu2 = -std::log10(std::abs(huMoments[1]));
    
    return features;
}

/**
 * Displays the region features on the output image
 * 
 * @param output The output image
 * @param features The region features to display
 */
void displayRegionFeatures(cv::Mat& output, const RegionFeatures& features) {
    cv::circle(output, features.center, 3, cv::Scalar(255, 255, 255), -1);
    
    double lineLength = 50.0;
    cv::Point2f endpoint1(
        features.center.x + lineLength * cos(features.orientation),
        features.center.y + lineLength * sin(features.orientation)
    );
    cv::Point2f endpoint2(
        features.center.x - lineLength * cos(features.orientation),
        features.center.y - lineLength * sin(features.orientation)
    );
    cv::line(output, endpoint1, endpoint2, cv::Scalar(0, 255, 255), 2);
    
    cv::Point2f rect_points[4];
    features.orientedBox.points(rect_points);
    for (int j = 0; j < 4; j++) {
        cv::line(output, rect_points[j], rect_points[(j+1)%4], cv::Scalar(0, 255, 0), 2);
    }
    
    std::string text = "Fill: " + std::to_string(int(features.percentFilled * 100)) + "%";
    cv::putText(output, text, cv::Point(features.center.x - 40, features.center.y - 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    std::string text2 = "AR: " + std::to_string(features.aspectRatio).substr(0, 4);
    cv::putText(output, text2, cv::Point(features.center.x - 40, features.center.y + 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

/**
 * Prints the feature vector for a region
 * 
 * @param features The region features to print
 * @param regionId The ID of the region
 */
void printFeatureVector(const RegionFeatures& features, int regionId) {
    std::cout << "Region " << regionId << " Features:" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Percent Filled: " << (features.percentFilled * 100) << "%" << std::endl;
    std::cout << "Aspect Ratio: " << features.aspectRatio << std::endl;
    std::cout << "Hu Moments: [" << features.hu1 << ", " << features.hu2 << "]" << std::endl;
    std::cout << "Center of Mass: (" << features.center.x << ", " << features.center.y << ")" << std::endl;
    std::cout << "Orientation: " << (features.orientation * 180.0 / CV_PI) << " degrees" << std::endl;
    std::cout << "--------------------------------" << std::endl;
}

/**
 * Struct to hold database entries
 * 
 * @param label The label of the object
 * @param features The feature vector of the object
 */
struct DatabaseEntry {
    std::string label;
    double features[4]; // percentFilled, aspectRatio, hu1, hu2
};

/**
 * Loads the database from a file
 * 
 * @param filename The name of the file to load the database from
 * @return A vector of database entries
 */
std::vector<DatabaseEntry> loadDatabase(const std::string& filename) {
    std::vector<DatabaseEntry> database;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        return database; 
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        DatabaseEntry entry;
        
        std::getline(ss, entry.label, ',');
        ss >> entry.features[0]; ss.ignore(); 
        ss >> entry.features[1]; ss.ignore(); 
        ss >> entry.features[2]; ss.ignore(); 
        ss >> entry.features[3];              
        
        database.push_back(entry);
    }
    
    file.close();
    return database;
}

/**
 * Calculates the standard deviations for the features in the database
 * 
 * @param database The database of objects
 * @return A vector of standard deviations
 */
std::vector<double> calculateStdDevs(const std::vector<DatabaseEntry>& database) {
    std::vector<double> stdDevs(4, 1.0); 
    
    if (database.size() <= 1) {
        return stdDevs; 
    }
    
    // Calculate means
    std::vector<double> sums(4, 0.0);
    std::vector<double> sumSquares(4, 0.0);
    
    for (const auto& entry : database) {
        for (int i = 0; i < 4; i++) {
            sums[i] += entry.features[i];
            sumSquares[i] += entry.features[i] * entry.features[i];
        }
    }
    
    // Calculate standard deviations
    for (int i = 0; i < 4; i++) {
        double mean = sums[i] / database.size();
        double variance = (sumSquares[i] / database.size()) - (mean * mean);
        stdDevs[i] = sqrt(variance);
        
        // Prevent division by zero
        if (stdDevs[i] < 0.0001) stdDevs[i] = 0.0001;
    }
    
    return stdDevs;
}

/**
 * Classifies an object with confidence using the distance metric
 * 
 * @param features The region features to classify
 * @param distanceMetric The distance metric to use (0: Euclidean, 1: Scaled Euclidean, 2: Cosine, 3: Scaled L1)
 * @return A pair containing the classification and confidence
 */
std::pair<std::string, double> classifyObjectWithConfidence(const RegionFeatures& features, int distanceMetric) {
    std::vector<DatabaseEntry> database = loadDatabase("data/object_features.csv");
    
    if (database.empty()) {
        return std::make_pair("Unknown (no database)", 0.0);
    }
    
    // Calculate standard deviations for normalization
    std::vector<double> stdDevs = calculateStdDevs(database);
    
    // Find nearest neighbor
    std::string bestMatch = "Unknown";
    double minDistance = DBL_MAX;
    
    for (const auto& entry : database) {
        double dist = 0.0;
        
        switch(distanceMetric) {
            case 0: // Simple Euclidean distance
                dist = sqrt(
                    pow(features.percentFilled - entry.features[0], 2) +
                    pow(features.aspectRatio - entry.features[1], 2) +
                    pow(features.hu1 - entry.features[2], 2) +
                    pow(features.hu2 - entry.features[3], 2)
                );
                break;
                
            case 1: // Scaled Euclidean distance (normalized by std dev)
                dist = sqrt(
                    pow((features.percentFilled - entry.features[0]) / stdDevs[0], 2) +
                    pow((features.aspectRatio - entry.features[1]) / stdDevs[1], 2) +
                    pow((features.hu1 - entry.features[2]) / stdDevs[2], 2) +
                    pow((features.hu2 - entry.features[3]) / stdDevs[3], 2)
                );
                break;
                
            case 2: // Cosine distance (1 - cosine similarity)
                {
                    double dotProduct = 
                        features.percentFilled * entry.features[0] +
                        features.aspectRatio * entry.features[1] +
                        features.hu1 * entry.features[2] +
                        features.hu2 * entry.features[3];
                        
                    double norm1 = sqrt(
                        pow(features.percentFilled, 2) +
                        pow(features.aspectRatio, 2) +
                        pow(features.hu1, 2) +
                        pow(features.hu2, 2)
                    );
                    
                    double norm2 = sqrt(
                        pow(entry.features[0], 2) +
                        pow(entry.features[1], 2) +
                        pow(entry.features[2], 2) +
                        pow(entry.features[3], 2)
                    );
                    
                    double similarity = dotProduct / (norm1 * norm2);
                    dist = 1.0 - similarity; 
                }
                break;
                
            case 3: // Scaled L1 (Manhattan) distance
                dist = 
                    fabs(features.percentFilled - entry.features[0]) / stdDevs[0] +
                    fabs(features.aspectRatio - entry.features[1]) / stdDevs[1] +
                    fabs(features.hu1 - entry.features[2]) / stdDevs[2] +
                    fabs(features.hu2 - entry.features[3]) / stdDevs[3];
                break;
        }
        
        if (dist < minDistance) {
            minDistance = dist;
            bestMatch = entry.label;
        }
    }
    
    double confidence = 0.0;
    double threshold = (distanceMetric == 2) ? 0.5 : 5.0; 
    
    if (minDistance < threshold) {
        confidence = 100.0 * (1.0 - minDistance/threshold);
        confidence = std::max(0.0, std::min(100.0, confidence)); // Clamp to 0-100%
    }
    
    // If confidence is too low, return Unknown
    if (confidence < 50.0) {
        return std::make_pair("Unknown", confidence);
    }
    
    return std::make_pair(bestMatch, confidence);
}

/**
 * Applies the thresholding operation to a frame
 * 
 * @param frame The source frame to apply the thresholding operation to
 * @param threshold_value The threshold value for the thresholding operation
 * @param morphOperation The type of morphological operation to perform (0: Erosion, 1: Dilation, 2: Opening, 3: Closing)
 * @param kernelSize The size of the kernel for the morphological operation (default is 3)
 * @param distanceMetric The distance metric to use (0: Euclidean, 1: Scaled Euclidean, 2: Cosine, 3: Scaled L1)
 * @return The output image with the regions colored
 */
cv::Mat applyThreshold(cv::Mat &frame, int threshold_value, int morphOperation, int kernelSize, int distanceMetric) {
    static std::vector<std::pair<cv::Point2d, cv::Vec3b>> regionTracker;
    
    cv::Mat blurred, hsv, gray, thresh, cleaned;
    
    blur5x5_2(frame, blurred);
    
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    
    cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);
    
    adjustGrayWithSaturation(gray, hsv_channels[1], 0.7f);
    
    thresh = ThresholdFunct(gray, threshold_value);
    
    cleaned = performMorphology(thresh, morphOperation, kernelSize);
    
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids);
    
    std::vector<int> validLabels = findValidRegions(stats, cleaned, 500, numLabels);
    
    //DEBUG:STATEMENT
    //std::cout << "Total regions: " << numLabels - 1 << ", Valid regions: " << validLabels.size() << std::endl;
    
    cv::Mat output = cv::Mat::zeros(cleaned.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors = generateColors(validLabels.size());
    
    std::vector<std::pair<cv::Point2d, cv::Vec3b>> currentRegions;
    
    assignRegionColors(validLabels, centroids, &regionTracker, colors, currentRegions);
    
    regionTracker = currentRegions;
    
    colorRegions(labels, validLabels, colors, output);
    
    g_labels = labels.clone();
    
    for(size_t i = 0; i < validLabels.size(); i++) {
        int label = validLabels[i];
        RegionFeatures features = computeRegionFeatures(labels, label);
        displayRegionFeatures(output, features);
        
        std::string metricName;
        switch(distanceMetric) {
            case 0: metricName = "Euclidean"; break;
            case 1: metricName = "Scaled Euclidean"; break;
            case 2: metricName = "Cosine"; break;
            case 3: metricName = "Scaled L1"; break;
        }
        
        auto classification = classifyObjectWithConfidence(features, distanceMetric);
        std::string objectClass = classification.first;
        double confidence = classification.second;
        
        // Display classification with confidence percentage and distance metric
        std::string displayText = objectClass + " (" + std::to_string(int(confidence)) + "%)";
        cv::putText(output, displayText, 
                    cv::Point(features.center.x - 60, features.center.y - 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    
        cv::putText(output, "Metric: " + metricName, 
                    cv::Point(features.center.x - 60, features.center.y - 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    
    return output;
}

/**
 * Saves the feature vector for an object
 * 
 * @param features The region features to save
 * @param label The label of the object
 */
void saveFeatureVector(const RegionFeatures& features, const std::string& label) {
    std::ofstream file("data/object_features.csv", std::ios::app); 
    if (!file.is_open()) {
        std::cerr << "Error: Could not open database file." << std::endl;
        return;
    }
    
    // Save the feature vector with its label
    file << label << ","
         << features.percentFilled << ","
         << features.aspectRatio << ","
         << features.hu1 << ","
         << features.hu2 << "\n";
    
    std::cout << "Saved feature vector for object: " << label << std::endl;
    file.close();
}
