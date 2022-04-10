#pragma once
#include "abstract_dataset_handler.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

class ImageDatasetHandler: public AbstractDatasetHandler<cv::Mat>{
    public:
        ImageDatasetHandler(const std::string root_folder): AbstractDatasetHandler(root_folder)
        {
            ImageDatasetHandler::loadDataset();
        };
        void loadDataset()
        {
            std::string root_folder = this->root_folder_;
            
            std::vector<std::string> folder_list;
            
            this->getFoldersFromPath(folder_list);
            
            std::list<cv::Mat> image_list;
            
            for(const auto & folder:folder_list)
            {
                getImagesFromFolder(folder,image_list);
                
                const std::string label_identifier = getLabelIdentifierFromFolderPath(folder);
                this->dataset_map_.insert(std::make_pair(label_identifier,image_list));
                std::cout<<"loading "<<folder<<"\n";
            }
        }
        void saveDataset(const std::string file_type,const std::string new_path="")
        {
           std::string folder_path = new_path;
           
           if(new_path.empty())
                   folder_path = root_folder_;
           else if(checkDirectory(new_path) == false){
                   fs::create_directory(new_path);
           }

           for(auto & component:dataset_map_)
           {

                const std::string label = component.first;
                
                const std::string child_folder_path  = folder_path+"/"+label;
                
                std::cout<<"saving "<<child_folder_path<<"\n";
                
                std::filesystem::create_directory(child_folder_path);

                int i = 0;
                
                for(const auto & img:component.second)
                {
                    std::string full_path_img = child_folder_path + "/"+std::to_string(i)+file_type;

                    cv::imwrite(full_path_img,img);
                    
                    i++;
                }   
            }

        }
    private:
        void getImagesFromFolder(const std::string folder_path, std::list<cv::Mat> & image_list)
        { 
            for (const auto & file  : std::filesystem::directory_iterator(folder_path))
            {
                if(isExtension(file.path(),"jpeg") || isExtension(file.path(),"jpg"))
                {
                    cv::Mat img;
                    const std::string path = file.path();
                    img = cv::imread(path);
                    cout << "loading picture  " << path << endl;
                    image_list.push_back(img);
                }
            }
        };
};
