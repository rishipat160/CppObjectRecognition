/**
 * Rishi Patel
 * Due: 02/20/2025
 * 
 * This was taken from Project 1 and modified to work with Project 3. 
 * Project 1 is originally a video display program that allows the user to toggle filters on and off
 * 
 * This code has been modified to work with Object Detection/Recognition.
 * Code is a bit messy, needs refactoring for sure.
 * 
 * The main key component of how a frame is processed and handled is still the same, along with key handling.
 */

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include "threshold.hpp"
#include <fstream>


extern cv::Mat g_labels;  

/**
 * 
 * @param argc The number of arguments passed to the program.
 * @param argv The arguments passed to the program.
 * @return The exit status of the program.
 */
int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;
        printf("Opening video device\n");
        capdev = new cv::VideoCapture(0 + cv::CAP_DSHOW);  

        if(!capdev->isOpened()) {
                printf("Unable to open video device\n");
                delete capdev;
                return(-1);
        }
        printf("Opened\n");
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); 
        cv::Mat frame;
        
        std::vector<std::string> objectLabels = {"mouse", "phone", "watch", "wallet", "tape"};
        int confusionMatrix[5][5] = {0}; // 5x5 matrix
        bool evaluationMode = false;
        std::string currentTrueLabel = "";
        int currentDistanceMetric = 0; // 0: Euclidean, 1: Scaled Euclidean, 2: Cosine, 3: Scaled L1

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }
                cv::Mat threshold = applyThreshold(frame, 100, 3, 3, currentDistanceMetric);

                cv::imshow("Video", frame);
                cv::imshow("Threshold", threshold);
                
                // Display current distance metric on the frame
                std::string metricText;
                switch(currentDistanceMetric) {
                    case 0: metricText = "Distance: Euclidean"; break;
                    case 1: metricText = "Distance: Scaled Euclidean"; break;
                    case 2: metricText = "Distance: Cosine"; break;
                    case 3: metricText = "Distance: Scaled L1"; break;
                }
                cv::putText(frame, metricText, cv::Point(10, 30), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

                char key = cv::waitKey(10);
                switch(key) {
                    case 'q': goto cleanup;  // quit
                    case 's': {  // save frame
                        static int count = 0;
                        cv::imwrite("image_" + std::to_string(count++) + ".jpg", frame);
                        cv::imwrite("threshold_" + std::to_string(count++) + ".jpg", threshold);
                        break;
                    }
                    case 'n': {  // save object features
                        if (g_labels.empty()) {
                            std::cout << "No valid regions found. Try again." << std::endl;
                            break;
                        }
                        
                        // Check if there are any non-zero labels in the image
                        bool foundValidRegion = false;
                        int objectRegion = 0;
                        
                        for (int i = 0; i < g_labels.rows && !foundValidRegion; i++) {
                            for (int j = 0; j < g_labels.cols && !foundValidRegion; j++) {
                                if (g_labels.at<int>(i, j) > 0) {
                                    objectRegion = g_labels.at<int>(i, j);
                                    foundValidRegion = true;
                                }
                            }
                        }
                        
                        if (!foundValidRegion) {
                            std::cout << "No valid regions found. Try again." << std::endl;
                            break;
                        }
                        
                        std::string label;
                        std::cout << "Enter label for this object: ";
                        std::cin >> label;
                        
                        // Compute features for this region
                        RegionFeatures features = computeRegionFeatures(g_labels, objectRegion);
                        
                        // Print the features
                        printFeatureVector(features, objectRegion);
                        
                        // Save to database
                        saveFeatureVector(features, label);
                        break;
                    } // All below are cases for Confusion Matrix
                    
                    case 'e': { // toggle evaluation mode
                        evaluationMode = !evaluationMode;
                        std::cout << "Evaluation mode: " << (evaluationMode ? "ON" : "OFF") << std::endl;
                        if (evaluationMode) {
                            std::cout << "Set true object label before testing:" << std::endl;
                            for (int i = 0; i < objectLabels.size(); i++) {
                                std::cout << i+1 << ": " << objectLabels[i] << std::endl;
                            }
                        }
                        break;
                    }
                    case '1': case '2': case '3': case '4': case '5': {
                        if (evaluationMode) {
                            int index = key - '1'; // Convert to 0-based index
                            currentTrueLabel = objectLabels[index];
                            std::cout << "True object set to: " << currentTrueLabel << std::endl;
                        }
                        break;
                    }
                    case 't': { // test current object and update confusion matrix
                        if (evaluationMode && !currentTrueLabel.empty()) {
                            if (g_labels.empty()) {
                                std::cout << "No valid regions found. Try again." << std::endl;
                                break;
                            }
                            
                            // Find first valid region
                            bool foundValidRegion = false;
                            int objectRegion = 0;
                            
                            for (int i = 0; i < g_labels.rows && !foundValidRegion; i++) {
                                for (int j = 0; j < g_labels.cols && !foundValidRegion; j++) {
                                    if (g_labels.at<int>(i, j) > 0) {
                                        objectRegion = g_labels.at<int>(i, j);
                                        foundValidRegion = true;
                                    }
                                }
                            }
                            
                            if (!foundValidRegion) {
                                std::cout << "No valid regions found. Try again." << std::endl;
                                break;
                            }
                            
                            // Compute features and classify
                            RegionFeatures features = computeRegionFeatures(g_labels, objectRegion);
                            auto classification = classifyObjectWithConfidence(features);
                            std::string predictedLabel = classification.first;
                            
                            // Find indices for the confusion matrix
                            int trueIndex = -1, predIndex = -1;
                            for (int i = 0; i < objectLabels.size(); i++) {
                                if (objectLabels[i] == currentTrueLabel) trueIndex = i;
                                if (objectLabels[i] == predictedLabel) predIndex = i;
                            }
                            
                            if (trueIndex >= 0 && predIndex >= 0) {
                                confusionMatrix[trueIndex][predIndex]++;
                                std::cout << "Recorded: True=" << currentTrueLabel 
                                          << ", Predicted=" << predictedLabel << std::endl;
                            }
                        }
                        break;
                    }
                    case 'p': { // print confusion matrix
                        std::cout << "\nConfusion Matrix:\n";
                        std::cout << "True\\Pred";
                        for (const auto& label : objectLabels) {
                            std::cout << "\t" << label;
                        }
                        std::cout << "\n";
                        
                        for (int i = 0; i < objectLabels.size(); i++) {
                            std::cout << objectLabels[i];
                            for (int j = 0; j < objectLabels.size(); j++) {
                                std::cout << "\t" << confusionMatrix[i][j];
                            }
                            std::cout << "\n";
                        }
                        break;
                    }
                    case 'd': { // cycle through distance metrics
                        currentDistanceMetric = (currentDistanceMetric + 1) % 4;
                        std::cout << "Distance metric changed to: " << metricText << std::endl;
                        break;
                    }
                }
        }

cleanup:
        // Clean up
        if(capdev != nullptr) {
            capdev->release();  // Release the camera
            delete capdev;      // Free the memory
            capdev = nullptr;
        }
        cv::destroyAllWindows();  // Close any OpenCV windows
        return(0);
}