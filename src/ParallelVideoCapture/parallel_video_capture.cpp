#include "parallel_video_capture.hpp"
#include <iostream>
ParallelVideoCapture::ParallelVideoCapture(int index, int fps) : cv::VideoCapture(index), running_(false), fps_(fps){
    interval_ms_ = 1000/fps;
}

ParallelVideoCapture::ParallelVideoCapture(const cv::String & filename, int fps) : cv::VideoCapture(filename), running_(false), fps_(fps){
    interval_ms_ = 1000/fps;
}

ParallelVideoCapture::ParallelVideoCapture(const cv::String &filename, int apiPreference, int fps) : cv::VideoCapture(filename,apiPreference), running_(false), fps_(fps){
    interval_ms_ = 1000/fps;
}

ParallelVideoCapture::~ParallelVideoCapture()
{
    stopCapture();
}


void ParallelVideoCapture::startCapture()
{
    if(isOpened())
    {
        running_ = true;
        is_capturing_ = false; // waits for grab the first frame to receive true

        ParallelVideoCapture*  ptr = static_cast<ParallelVideoCapture*> (this);
        
        if(ptr == nullptr)
        {
            throw std::logic_error("[ERROR][startCapture()] error to convert this object into ParallelVideoCapture* ptr");
        }

        thread_ptr_ = std::make_unique<std::thread>(&captureFromSource,std::ref(*ptr));

        waitForCapture();  
    }
    else{
        throw std::ios_base::failure("[ERROR][startCapture()] could not to open the source video\n");
    }

}

bool ParallelVideoCapture::isRunning() const
{
    return running_;
}

bool ParallelVideoCapture::isOpened() const
{
    bool opened = cv::VideoCapture::isOpened();
    
    return opened;
}

uint8_t ParallelVideoCapture::getIntervalMs() const
{
    return interval_ms_;
}

void ParallelVideoCapture::stopCapture()
{
    is_capturing_ = false;

    if(running_)
    {
        running_ = false;
        // we join the thread to sync it 
        // this will make out stopCapture function wait for the thread exits from it loop
        thread_ptr_->join();    
    }

}
bool ParallelVideoCapture::read()
{
    mutex_.lock();
        bool readed = read(frame_);
    mutex_.unlock();
    
    return readed;
}

bool ParallelVideoCapture::read(cv::OutputArray image)
{
    return  cv::VideoCapture::read(image);
}

bool ParallelVideoCapture::grab()
{
    bool readed = cv::VideoCapture::grab();
    
    is_capturing_ = readed;
    
    return readed;
}
bool ParallelVideoCapture::retrieve(cv::OutputArray image, int flag)
{
    bool readed = cv::VideoCapture::retrieve(image,flag);
    
    return readed;
}
bool ParallelVideoCapture::retrieve(int flags)
{
    mutex_.lock();
        bool readed = cv::VideoCapture::retrieve(frame_,flags);
    mutex_.unlock();

    return readed;
}
void ParallelVideoCapture::release()
{
    cv::VideoCapture::release();
}
void ParallelVideoCapture::getFrame(cv::Mat & frame)
{
    mutex_.lock();
        frame = frame_.clone(); // we must to copy the image to avoid non deterministic situations
    mutex_.unlock();
}
cv::Mat ParallelVideoCapture::getFrame()
{
    cv::Mat img;  

    getFrame(img); 

    return img;

}
void captureFromSource(ParallelVideoCapture & cap)
{
    bool readed = false;
    
    auto start = std::chrono::system_clock::now();
    auto end  = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout<<"[INFO] starting capture thread\n";
    while(cap.isRunning())
    {
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        auto time_dif = elapsed.count()-cap.getIntervalMs();
        if(time_dif<0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(abs(time_dif)));

            end  = std::chrono::system_clock::now();
            continue;
        }
        start = std::chrono::system_clock::now();

        readed = cap.read();

        if(!readed)
            break;
        
    }

    cap.release();

}
