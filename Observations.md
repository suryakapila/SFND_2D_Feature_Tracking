# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Observations from the Project
## [Rubric](https://review.udacity.com/#!/rubrics/2549/view) Points



### MP1. DATA BUFFER

##  MP.1 Data Buffer Optimization
* Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end.
* This is achieved by checking the size of DataBuffer doesnot exceed DataBuffer size- once the DataBuffer reaches its limit of 3 elements the new image is pushed into DataBuffer and the oldest image is erased.
```c++
DataFrame frame;
            frame.cameraImg = imgGray;
            cout<< dataBuffer.size()<<"check_size"<<endl;
            if(dataBuffer.size() > dataBufferSize)
            {
                dataBuffer.erase(dataBuffer.begin());
            }
            dataBuffer.push_back(frame);
```

### 2. KEYPOINT DETECTION IMPLEMENTATION
 
## MP.2 Keypoint Detection
* Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.
* Solution is implemented in multiple locations:
1. Calling the functions into `MidTermProject_Camera_Student.cpp`
```c++
 //// STUDENT ASSIGNMENT
            //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType


            if (detectorType.compare("SHITOMASI") == 0)
            {
                detKeypointsShiTomasi(keypoints, imgGray, false);
            }
          ////// //// -> HARRIS
            else if (detectorType.compare("HARRIS")==0)
            {
                detKeypointsHarris(keypoints, imgGray, false);
            }
          //// ->FAST, BRISK, ORB, AKAZE, SIFT
            else if (detectorType.compare("FAST")==0||
                    detectorType.compare("BRISK")==0||
                    detectorType.compare("ORB")==0||
                    detectorType.compare("AKAZE")==0||
                    detectorType.compare("SIFT")==0)
            {
                detKeypointsModern(keypoints, imgGray, detectorType, false ); 
            }

            else
              cerr<<"the given detectortype doesnot exist here, try a valide one" <<endl;

            //// EOF STUDENT ASSIGNMENT
 ```
 2.Defining the detectors in  `matching2D_Student.cpp`
  ```c++
 ///Implementing HARRIS Keypoint Detector
 void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
  
  // Look for prominent corners and instantiate keypoints
   // vector<cv::KeyPoint> keypoints; //keypoints instantiated
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof
  
    t = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
    cout << "HARRIS Corner detection with n ="<< keypoints.size()<< "keypoints in" << 1000*t/1 <<"ms" << endl;
	// visualize results
    if (bVis)
    {
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "HARRIS Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
 
 //// Implemetation of modern Detectors ->FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
   double t1 = (double)cv::getTickCount();
   cv::Ptr<cv::Feature2D> detector;     // Reference : https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
  //implementation of FAST detector algorithm
   if(detectorType.compare("FAST")==0)
   {
     int threshold = 30; //difference between the intensity of the centre pixel and the sorrouding pixels
     bool bNMS = true; //perform Non-Maxima Suppression on keypoints
     
     detector = cv::FastFeatureDetector::create(threshold, bNMS, cv::FastFeatureDetector::TYPE_9_16);
     
   }
  //implementation of BRISK detector
   else if(detectorType.compare("BRISK")==0)
   {
    detector = cv::BRISK::create();
    
   }
   else if(detectorType.compare("ORB")==0)
   {
    detector = cv::ORB::create();
    
   }
   else if(detectorType.compare("AKAZE")==0)
   {
    detector = cv::AKAZE::create();
    
   }
   else if(detectorType.compare("SIFT")==0)
   {
    detector = cv::xfeatures2d::SIFT::create();
    
   }
   detector->detect(img, keypoints);
   t1 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
   cout << detectorType<< "with n= " << keypoints.size() << " keypoints in " << 1000 * t1 / 1.0 << " ms" << endl;
  
   // visualize results
   if (bVis)
   {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + "Detector Keypoints Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
   }
  
}

 ```
 
 ### 3.KEYPOINT REMOVAL
 ## MP.3 Keypoint Removal
 * Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.
 
 
 ```c++
             //// STUDENT ASSIGNMENT
            //// TASK MP.3 -> only keep keypoints on the preceding vehicle

            // only keep keypoints on the preceding vehicle
            bool bFocusOnVehicle = true;
            cv::Rect vehicleRect(535, 180, 180, 150);
            if (bFocusOnVehicle)
            {
              vector<cv::KeyPoint> kptsROI; //instantiating a new container to store keypoint values in ROI
              for(auto it = keypoints.begin(); it!=keypoints.end();++it)
              {
               if(vehicleRect.contains(it->pt)) //check for keypoints in ROI
               {
                /////cannot directly copy keypoints in ROI to a new vector ,
                /// so instantiate  keypoint variable and assign the keypoint coordinates(point2f) to this 
                ///and push_back this keypoint into the vector ROI

                cv::KeyPoint keypoint_xy;
                keypoint_xy.pt = cv::Point2f(it->pt);
				keypoint_xy.size = 1;
                kptsROI.push_back(keypoint_xy);
               }
              }
             keypoints = kptsROI;
             cout << "Num of Keypoints in ROI is n= "<< keypoints.size()<<endl;
            }
            ///Tip:alternatively can also delete all the keypoints that are not within !vehicleRect using erase function
            //// EOF STUDENT ASSIGNMENT TASK MP.3
```

### 4.Keypoint Descriptors
## MP.4 Keypoint Descriptors
* Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.
* Solution to this task is also in Multiple files
1. Instantiation of the Keypoint descriptors  in `MidTermProject_Camera_Student.cpp`

```c++
//// STUDENT ASSIGNMENT
            //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
            //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

            cv::Mat descriptors;
            string descriptor_Type = ds; // BRIEF, ORB, FREAK, AKAZE, SIFT
            descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptor_Type);
            //// EOF STUDENT ASSIGNMENT
```
2. Definition/Implemenation of these keypoint dexcriptors is in `matching2D_Student.cpp`

```c++
// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
  //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
    else if (descriptorType.compare("BRIEF")==0)
    {
      extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();      
    }
    else if (descriptorType.compare("ORB")==0)
    {
      extractor = cv::ORB::create();
    }
  	else if (descriptorType.compare("FREAK")==0)
    {
      extractor = cv::xfeatures2d::FREAK::create();
    }
	else if (descriptorType.compare("AKAZE")==0)
    {
      extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT")==0)
    {
      extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    
    }
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

    
} //End of TASK MP.4

```
### 5. Descriptor Matching

## MP.5 Descriptor Matching
* Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.
* Solution is implemented in the file `matching2D_Student.cpp`
```c++

    if (matcherType.compare("MAT_BF") == 0)
    {
       int normType =  cv::NORM_L2;
       if(descriptorType.compare("DES_BINARY")==0)
       {
         normType = cv::NORM_HAMMING;
       }
       matcher = cv::BFMatcher::create(normType, crossCheck);
    }
  //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
         if (descSource.type() != CV_32F || descRef.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        //... TODO : implement FLANN matching
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED); 
        cout << "FLANN matching"<< endl;
    }///EOF student TASK MP.5


```
### 6.Descriptor Distance Ratio

## MP.6 Descriptor Distance Ratio
* Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.
* Solution is impplemeted in multiple flies
1. The method KNN matcher is selected in `MidTermProject_Camera_Student.cpp`
```c++
 //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering  with t=0.8 in file matching2D.cpp(done)
                string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
```
2. The implmentation of KNN with distance ratio filtering is in  `matching2D_Student.cpp`
```c++
 //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
       vector<vector<cv::DMatch>> knn_matches;
            // TODO : implement k-nearest-neighbor matching
        
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        

        // TODO : filter matches using descriptor distance ratio test
        double minDescDisRatio = 0.8;
        for(auto it = knn_matches.begin(); it!= knn_matches.end(); ++it)
        {
            if((*it)[0].distance < minDescDisRatio*(*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout<<"#keypoints removed = "<< knn_matches.size()- matches.size() <<endl;
		
    }//End of TASK MP.6
    
 ```
 #### PERFORMANCE
 
 ### 7. Performance Evaluation 1 - Distribution of Neighbourhood
 
 ## MP.7 Performance Evaluation 1
 * Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size.
 * Do this for all the detectors you have implemented.
 * Solution is stored in the .csv files in this project `MP7_Count_Keypoints.csv`file .[CSV file].(https://github.com/suryakapila/SFND_2D_Feature_Tracking/blob/master/MP7_Count_Keypoints.csv).
 * The distribution of neighbourhood of keypoints of various Kepoint detectors is as follows:
 
 Detector  | Neighbour of keypoints
-----------|----------------------
 SHITOMASI |  111 ~ 125             
 HARRIS    |  14 ~ 43                
 ORB       |  92 ~ 130               
 FAST      |  138 ~ 156              
 BRISK     |  254 ~ 297              
 AKAZE     |  155 ~ 179             
 SIFT      |  124 ~ 159              

* As seen in the table above, HARRIS Keypoint Detector has least number of keypoints detected among the set of detectors and BRISK has highest number of keypoints detected. 
 
### 8. Matched Keypoints

## MP.8 Performance Evaluation 2

* Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. 
* In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.
* The number of matched keypoints can be seen in the file `MP8_CountMatches.csv`.[CSVfile].(https://github.com/suryakapila/SFND_2D_Feature_Tracking/blob/master/MP8_CountMatches.csv)

### 9.Top3 Detector/Descriptor Matching

## MP.9 Performance Evaluation 3
	
* Log the time it takes for keypoint detection and descriptor extraction. 
* The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.

RANK | Detector+Descriptor | Time taken | Avg keypoints
-----|---------------------|------------|----------------
  1  |  FAST+BRIEF         | 5.69 ms    | 122
  2  |  FAST+ORB           | 5.73 ms    | 118
  3  |  ORB+BRIEF          | 7.91 ms    | 66
  
 * Based on the average number of keypoints and time taken for the implementation of Detector+Descriptor combination, these are the three top performers chosen as the best choice for our purpose of detecting keypoints on vehicles.


*Special mention to `https://github.com/studian/SFND_P3_2D_Feature_Tracking`whose `writeup.md` is the inspiration for this file*
 
 




