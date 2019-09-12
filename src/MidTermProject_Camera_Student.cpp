/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */
     
   
     // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */
   ///////implementation of write files using ofstream and automating the test process
    ////// Implementation of additional tasks MP7 , MP8 & MP9
  	vector<string> detector_type_name = {"SHITOMASI", "HARRIS", "ORB", "FAST", "BRISK", "AKAZE", "SIFT" };
    vector<string> descriptor_type_name = {"BRIEF", "BRISK","FREAK", "ORB", "AKAZE", "SIFT"}; 
    ofstream detector_results;
    detector_results.open("../MP7_Count_Keypoints.csv");
    
    ofstream descriptor_matches;
    descriptor_matches.open("../MP8_CountMatches.csv");
  
    ofstream time_top3;
    time_top3.open("../MP9_Log_Time.csv");
  
  ////Run through the whole program for writing the results into a file
    for(auto dt : detector_type_name)
    {
     bool writer = true;
     for(auto ds:descriptor_type_name)
     {
       //AKAZE is not a friendly descriptor to other detectors & causes abrupt halt in program
      if(dt.compare("AKAZE")!=0 && ds.compare("AKAZE")==0)
        continue;
      if(dt.compare("AKAZE")==0 && ds.compare("AKAZE")==0)
         continue;
       dataBuffer.clear();
       cout << "detector type : "<< dt <<"descriptor type :" << ds << endl;
       //start TASK MP.7 implementation
       if(writer)
         detector_results << dt;
       //.....en TASK MP.7 implementation
       
       ////...start of TASK MP8. implementation
       descriptor_matches << dt << "+" << ds;
       //...end of TASK MP8
       
       //....start of TASK MP9. implementation
       time_top3 << dt << "+" << ds ;
       //...end of TASK MP9
        //// ***LOOP THROUGH ALL IMAGES ***
        for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
        {
            /* LOAD IMAGE INTO BUFFER */

            // assemble filenames for current index
            ostringstream imgNumber;
            imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
            string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

            // load image from file and convert to grayscale
            cv::Mat img, imgGray;
            img = cv::imread(imgFullFilename);
            cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

            //// STUDENT ASSIGNMENT
            //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

            // push image into data frame buffer
            DataFrame frame;
            frame.cameraImg = imgGray;
            cout<< dataBuffer.size()<<"check_size"<<endl;
            if(dataBuffer.size() > dataBufferSize)
            {
                dataBuffer.erase(dataBuffer.begin());
            }
            dataBuffer.push_back(frame);

            //// EOF STUDENT ASSIGNMENT
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

            /* DETECT IMAGE KEYPOINTS */
            // ... start TASK MP9 implementation
			double t = (double)cv::getTickCount();
            // extract 2D keypoints from current image
            vector<cv::KeyPoint> keypoints; // create empty feature list for current image
            string detectorType = dt; //try SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

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

                kptsROI.push_back(keypoint_xy);
               }
              }
             keypoints = kptsROI;
             cout << "Num of Keypoints in ROI is n= "<< keypoints.size()<<endl;
            }
            ///Tip:alternatively can also delete all the keypoints that are not within !vehicleRect using erase function
            //// EOF STUDENT ASSIGNMENT TASK MP.3
          
            //start TASK MP.7 implementation
      		 if(writer)
               detector_results << ","<< keypoints.size();
            //.....en TASK MP.7 implementation

            // optional : limit number of keypoints (helpful for debugging and learning)
            bool bLimitKpts = false;
            if (bLimitKpts)
            {
                int maxKeypoints = 50;

                if (detectorType.compare("SHITOMASI") == 0)
                { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                    keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                }
                cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                cout << " NOTE: Keypoints have been limited!" << endl;
            }

            // push keypoints and descriptor for current frame to end of data buffer
            (dataBuffer.end() - 1)->keypoints = keypoints;
            cout << "#2 : DETECT KEYPOINTS done" << endl;

            /* EXTRACT KEYPOINT DESCRIPTORS */

            //// STUDENT ASSIGNMENT
            //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
            //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

            cv::Mat descriptors;
            string descriptor_Type = ds; // BRIEF, ORB, FREAK, AKAZE, SIFT
            descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptor_Type);
            //// EOF STUDENT ASSIGNMENT

            // push descriptors for current frame to end of data buffer
            (dataBuffer.end() - 1)->descriptors = descriptors;

            cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

            if (dataBuffer.size() > 1) // wait until at least two images have been processed
            {

                /* MATCH KEYPOINT DESCRIPTORS */

                vector<cv::DMatch> matches;
                string matcherType = "MAT_BF";     // MAT_BF, MAT_FLANN
                string descriptorType = "DES_HOG";               // DES_BINARY, DES_HOG
                if(descriptor_Type.compare("SIFT")!=0)
                {
                 descriptorType = "DES_BINARY"; 
                }
                
                
                string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

                //// STUDENT ASSIGNMENT
                //// TASK MP.5 -> add FLANN matching in file matching2D.cpp (done)
                //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering  with t=0.8 in file matching2D.cpp(done)

                matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                 (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                 matches, descriptorType, matcherType, selectorType);

                //// EOF STUDENT ASSIGNMENT

                // store matches in current data frame
                (dataBuffer.end() - 1)->kptMatches = matches;

                cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
				////...start of TASK MP8. implementation
    			   descriptor_matches << "," << matches.size();
       			//...end of TASK MP8
              
               //....start of TASK MP9. implementation
               t = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
               time_top3 << "," << 1000*t /1.0 << "ms" ;
               //...end of TASK MP9
                
              
              
                // visualize matches between current and previous image
                bVis =true;
               cout << "Visualising matches between current and previous image"<<endl;
                if (bVis)
                {
                    cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                    cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                    (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                    matches, matchImg,
                                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                                    vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    //cv::drawKeypoints((dataBuffer.end()-2)->cameraImg,(dataBuffer.end()-2)->keypoints, matchImg, cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    string windowName = "Matching keypoints between two camera images";
                    cv::namedWindow(windowName, 7);
                    cv::imshow(windowName, matchImg);
                    cout << "Press key to continue to next image" << endl;
                    cv::waitKey(0); // wait for key to be pressed
                }

                bVis = false;
            }

        } // eof loop over all images
       //start TASK MP.7 implementation
       if(writer)
         detector_results << endl;
       //.....end TASK MP.7 implementation
       writer = false;
         descriptor_matches << endl;
         time_top3 <<endl;
     }//eof over descriptor types
   }//eof loop over detector types
       
       detector_results.close();
       descriptor_matches.close();
       time_top3.close();
    return 0;
}//eof of main
