#pragma once

#include "abstract_dataset_handler.hpp"
#include <exception>
#include <ios>
#include <iostream>
#include <dlib/image_io.h>

#include <fstream>

typedef cv::Mat  CV_DESCRIPTOR;

class DescriptorDatasetHandler: public AbstractDatasetHandler<CV_DESCRIPTOR>
{
    public:

       DescriptorDatasetHandler(const std::string root_folder):AbstractDatasetHandler(root_folder)
       {
           DescriptorDatasetHandler::loadDataset();
       }

        void loadDataset()
        {
            cout << "loading dataset" << endl;
            std::string root_folder = this->root_folder_;
            
            std::vector<std::string> folder_list;
            
            this->getFoldersFromPath(folder_list);
            
            std::list<DLIB_FLOAT_DESCRIPTOR> descriptor_list;
            
            for(const auto & folder:folder_list)
            {
                getDescriptorsFromFolder(folder,descriptor_list);
                
                const std::string label_identifier = getLabelIdentifierFromFolderPath(folder);
                this->dataset_map_.insert(std::make_pair(label_identifier,descriptor_list));
                std::cout<<"loading "<<folder<<"\n";
            }

        }
        void saveDataset(const std::string file_type,const std::string new_path="")
        {
           std::string folder_path = new_path;
           std::cout<<"root path "<<root_folder_<<"\n";

           if(new_path.empty())
                   folder_path = root_folder_;
           
           for(auto & component:dataset_map_)
           {

                const std::string label = component.first;
                
                const std::string child_folder_path  = folder_path+"/"+label;
                
                std::cout<<"saving "<<child_folder_path<<"\n";
                
                std::filesystem::create_directory(child_folder_path);

                int i = 0;
                
                for(const auto & descriptor:component.second)
                {
                    std::string full_path_descriptor = child_folder_path + "/"+std::to_string(i)+file_type;

                    saveDescriptor(descriptor,full_path_descriptor);
                    
                    i++;
                }   
            }

        }
        
    private:
        void loadDescriptor(DLIB_FLOAT_DESCRIPTOR & descriptor,const std::string filepath)
        {
                std::ifstream file(filepath,std::ios_base::binary); 
                if(file.is_open())
                {
                    std::cout<<"reading the file "<<filepath<<"\n";
                    file>>descriptor;  
                }
                else{
                    std::cout<<"file "<<filepath<<" not found\n";
                    return;
                }

        }
        void saveDescriptor(const CV_DESCRIPTOR & descriptor,const std::string filepath)
        {
            std::ofstream file(filepath,std::ios_base::binary);
            if(file.is_open())
            {
               file<<descriptor;   
            }
             else{
                     return;
              }

        }

        void getDescriptorsFromFolder(const std::string folder_path, std::list<DLIB_FLOAT_DESCRIPTOR> & descriptor_list)
        { 
            cout << "Getting Descriptor" << endl;
            cout << "folder path " << folder_path << endl;
            for (const auto & file  : std::filesystem::directory_iterator(folder_path))
            {
                cout << file.path() << endl;
                if(isExtension(file.path(),"data"))
                {
                    cout << "loading data" << file.path() << "\n";
                    DLIB_FLOAT_DESCRIPTOR descriptor;
              
                    loadDescriptor(descriptor,file.path());  
                    descriptor_list.push_back(descriptor);
                }
                else cout << "no data file found!" << file.path() << "\n";
            }
        };
      
};
