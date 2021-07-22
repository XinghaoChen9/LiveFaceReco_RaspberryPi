#pragma once
#include <filesystem>
#include <string>
#include <list>
#include <utility>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <map>
#include <iterator>
namespace fs  = std::filesystem;


using namespace std;

template<class T>
class AbstractDatasetHandler
{   
    public:
        
        AbstractDatasetHandler(AbstractDatasetHandler & ds_handler)
        {
            this->root_folder_ = ds_handler.root_folder_;
            this->dataset_map_ = ds_handler.dataset_map_;
        }

        std::map<std::string,std::list<T>>  & getDatasetMap()
        {
            return dataset_map_;
        }
        virtual void loadDataset() = 0;
        virtual void saveDataset(const std::string file_type, std::string target_path) = 0;
     
        //this methods check if a path is a directory
        // if not it throws an exception
        const static bool checkDirectory(const std::string & path)
        {
        
            fs::directory_entry d (path.c_str());
            if(fs::exists(d))
            {
                if(!fs::is_directory(d)) // if already exists and is not a directory this path is invalid
                {
                    const std::string msg = path + "is not a directory\n";
                    throw std::invalid_argument(msg.c_str());
                }
                return true;
            }
            return false;
        };
       
        void addComponent(const std::string & label, std::list<T> & data)
        {
            dataset_map_.insert(std::make_pair(label,data));
        };
        void addComponent(const std::string & label, std::vector<T> & data)
        {
            std::list<T> dest(data.begin(), data.end());
            dataset_map_.insert(std::make_pair(label,dest));
        };
        typename map<string,list<T>>::iterator begin()
        {
            return dataset_map_.begin();
        };
        typename map<string,list<T>>::iterator end()
        {
            return dataset_map_.end();
        };
    protected:
        std::string root_folder_;
        // each person is represented by a name and a list of images;
        std::map<std::string,std::list<T>> dataset_map_;

        AbstractDatasetHandler(const std::string & root_folder)
        {                
            if(!checkDirectory(root_folder)) 
            {   
                /**
                 *  If the directory doenst exists
                 *  we need to create a new directory
                 */
                fs::create_directory(root_folder);
            }
           
            root_folder_ = root_folder;
        }

        void getFoldersFromPath (std::vector<std::string> & folder_list)
        {
            for (const auto & file  : std::filesystem::recursive_directory_iterator(root_folder_))
            {
                if(std::filesystem::is_directory(file))
                {
                    folder_list.push_back(file.path());
                }
            }
        };
        bool isExtension(const std::string filename, const std::string suffix)
        {
            std::size_t found = filename.find_last_of('.')+1;
            std::string extension =  filename.substr(found,suffix.size());
            return extension == suffix;
        }
        const std::string getLabelIdentifierFromFolderPath(std::string folder_path)
        {
            int last_sepatator_pos = folder_path.find_last_of('/');
            std::string aux = folder_path.substr(last_sepatator_pos+1,folder_path.size());
            return aux;
        }
                
};


