//
// Created by markson zhang
//
//
// Edited by Xinghao Chen 2020/7/27
//
//
// Refactored and edited by Luiz Correia 2021/06/20


#include <math.h>
#include "livefacereco.hpp"
#include <time.h>
#include "math.hpp"
#include "ParallelVideoCapture/parallel_video_capture.hpp"
#include "mtcnn_new.h"
#include "FacePreprocess.h"

#define PI 3.14159265
using namespace std;

double sum_score, sum_fps,sum_confidence;

#define  PROJECT_PATH "/home/luiz/Faculdade/TCC/LiveFaceReco_RaspberryPi";

void calculateFaceDescriptorsFromDisk(Arcface & facereco,std::vector<cv::Mat> & face_descriptors)
{
    const std::string project_path = PROJECT_PATH;
    std::string pattern_jpg = project_path + "/img/*.jpg";
	std::vector<cv::String> image_names;
    
	cv::glob(pattern_jpg, image_names);

    int image_number=image_names.size();

	if (image_number == 0) {
		std::cout << "No image files[jpg]" << std::endl;
        return;
	}
    cout <<"loading pictures..."<<endl;
    
    cv::Mat  face_img;
    unsigned int img_idx = 0;

  
    //convert to vector and store into fc, whcih is benefical to furthur operation
	for(auto const & img_name:image_names)
    {
        face_img = cv::imread(img_name);

        cv::Mat face_descriptor = facereco.getFeature(face_img);
       
        face_descriptors.push_back(Statistics::zScore(face_descriptor));

        printf("\rloading[%.2lf%%]\n",  (++img_idx)*100.0 / (image_number));
    }
   
    cout <<"loading succeed! "<<image_number<<" pictures in total"<<endl;
    
}

void loadLiveModel( Live & live )
{
    //Live detection configs
    struct ModelConfig config1 ={2.7f,0.0f,0.0f,80,80,"model_1",false};
    struct ModelConfig config2 ={4.0f,0.0f,0.0f,80,80,"model_2",false};
    vector<struct ModelConfig> configs;
    configs.emplace_back(config1);
    configs.emplace_back(config2);
    live.LoadModel(configs);
}
cv::Mat createFaceLandmarkGTMatrix()
{
    // groundtruth face landmark
    float v1[5][2] = {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}};

    cv::Mat src(5, 2, CV_32FC1, v1); 
    memcpy(src.data, v1, 2 * 5 * sizeof(float));
    return src.clone();
}
cv::Mat createFaceLandmarkMatrixfromBBox(const Bbox  & box)
{
    float v2[5][2] =
                {{box.ppoint[0], box.ppoint[5]},
                {box.ppoint[1], box.ppoint[6]},
                {box.ppoint[2], box.ppoint[7]},
                {box.ppoint[3], box.ppoint[8]},
                {box.ppoint[4], box.ppoint[9]},
                };
    cv::Mat dst(5, 2, CV_32FC1, v2);
    memcpy(dst.data, v2, 2 * 5 * sizeof(float));

    return dst.clone();
}

Bbox  getLargestBboxFromBboxVec(const std::vector<Bbox> & faces_info)
{
    if(faces_info.size()>0)
    {
        int lagerest_face=0,largest_number=0;
        for (int i = 0; i < faces_info.size(); i++){
            int y_ = (int) faces_info[i].y2 * ratio_y;
            int h_ = (int) faces_info[i].y1 * ratio_y;
            if (h_-y_> lagerest_face){
                lagerest_face=h_-y_;
                largest_number=i;                   
            }
        }
        
        return faces_info[largest_number];
    }
    return Bbox();
}

LiveFaceBox Bbox2LiveFaceBox(const Bbox  & box)
{
    float x_   =  box.x1;
    float y_   =  box.y1;
    float x2_ =  box.x2;
    float y2_ =  box.y2;
    int x = (int) x_ ;
    int y = (int) y_;
    int x2 = (int) x2_;
    int y2 = (int) y2_;
    struct LiveFaceBox  live_box={x_,y_,x2_,y2_} ;
    return live_box;
}

cv::Mat alignFaceImage(const cv::Mat & frame, const Bbox & bbox,const cv::Mat & gt_landmark_matrix)
{
    cv::Mat face_landmark = createFaceLandmarkMatrixfromBBox(bbox);

    cv::Mat transf = FacePreprocess::similarTransform(face_landmark, gt_landmark_matrix);

    cv::Mat aligned = frame.clone();
    cv::warpPerspective(frame, aligned, transf, cv::Size(96, 112), INTER_LINEAR);
    resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);
     
    return aligned.clone();
}

int getClosestFaceDescriptorIdx(std::vector<cv::Mat> disk_face_descriptors, cv::Mat face_descriptor)
{
    vector<double> score_(disk_face_descriptors.size());
    int i = 0;
    for(const auto & disk_descp:disk_face_descriptors)
    {
        score_[i] = (Statistics::cosineDistance(disk_descp, face_descriptor));
        i++;
    }
    int maxPosition = max_element(score_.begin(),score_.end()) - score_.begin(); 
    int pos = score_[maxPosition]>distance_threshold?maxPosition:-1;
    score_.clear();
    return pos;
}

int MTCNNDetection()
{
    //OpenCV Version
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "."
    << CV_MINOR_VERSION << "."
    << CV_SUBMINOR_VERSION << endl;

    Arcface facereco;
    std::vector<cv::Mat> disk_face_descriptors;
    calculateFaceDescriptorsFromDisk(facereco,disk_face_descriptors);

    Live live;
    loadLiveModel(live);

    float factor = 0.709f;
    float threshold[3] = {0.7f, 0.6f, 0.6f};

    ParallelVideoCapture cap("udpsrc port=5000 ! application/x-rtp, payload=96 ! rtpjitterbuffer ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false",cv::CAP_GSTREAMER,30); //using camera capturing
                           
    cap.startCapture();

    std::cout<<"okay!\n";

    if (!cap.isOpened()) {
        cerr << "cannot get image" << endl;
        return -1;
    }

    float confidence;
    vector<float> fps;
    static double current;
    static char string[10];
    static char string1[10];
    char buff[10];
    Mat frame;
    Mat result_cnn;

    double score, angle;

    cv::Mat face_landmark_gt_matrix = createFaceLandmarkGTMatrix();
    int count = -1;
    std::string liveface;
    float ratio_x = 1;
    float ratio_y = 1;

    while(cap.isOpened())
    {
        frame = cap.getFrame();    
        //cv::resize(frame,frame,cv::Size(300,300));

        if(frame.empty())
        {
            continue;
        } 
        ++count;

        //detect faces
        std::vector<Bbox> faces_info = detect_mtcnn(frame); 
        if(faces_info.size()>=1)
        {

            auto large_box = getLargestBboxFromBboxVec(faces_info);
            LiveFaceBox live_face_box = Bbox2LiveFaceBox(large_box);
            
            cv::Mat aligned_img = alignFaceImage(frame,large_box,face_landmark_gt_matrix);

            cv::Mat face_descriptor = facereco.getFeature(aligned_img);
            // normalize
            face_descriptor = Statistics::zScore(face_descriptor);

            int idx = getClosestFaceDescriptorIdx(disk_face_descriptors,face_descriptor);
            
            if(idx>=0)
            {
                cout<<"face conhecida\n";
            }
            else{
                cout<<"face estranha\n";
            }

            confidence = live.Detect(frame,live_face_box);

            if (confidence<=true_thre)
            {
                //putText(result_cnn, "Fake face!!", cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 0, 255),2);
                liveface="Fake face!!";
            }
            else
            {
                //putText(result_cnn, "True face", cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(255, 255, 0),2);     
                liveface="True face";
            }

            cout<<liveface<<"\n";
            cv::putText(frame,liveface,cv::Point(15,40),1,2.0,cv::Scalar(255,0,0));
            cv::rectangle(frame, Point(large_box.x1*ratio_x, large_box.y1*ratio_y), Point(large_box.x2*ratio_x,large_box.y2*ratio_y), cv::Scalar(0, 0, 255), 2);
        }
       
        cv::imshow("img", frame);

        char k = cv::waitKey(33);
    
        if(k == 27)
            break;
        count ++;
    }
    cap.stopCapture();
    return 0;
}