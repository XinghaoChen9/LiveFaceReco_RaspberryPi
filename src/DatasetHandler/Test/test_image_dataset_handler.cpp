#include <iostream>
#include <dataset_handler.hpp>
#include <stdexcept>
using namespace std;

int main(int argc, char ** argv)
{

    try
    {
        if(argc!=2)
        {
            throw std::invalid_argument("usage: ./test_image_dataset_handler <dataset_path>");
        }
        
        ImageDatasetHandler dataset = ImageDatasetHandler(argv[1]);
        
        dataset.saveDataset(".jpg","DatasetImage");

    }
    catch(std::invalid_argument e){
        std::cout<<e.what()<<"\n";
    }
    
    return 0;

}
