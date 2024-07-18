#define debug

#ifdef debug 
#define LOG(msg) std::cout<<"LOG: " << msg << std::endl
#else
#define LOG(msg)
#endif


#include <algorithm>
#include <vector>
#include <fstream>
#include <math.h>
#include <string>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <thread>

#include <neural/activations.hpp>
#include <neural/math.hpp>
#include <neural/filehandle.hpp>
#include <neural/network.hpp>
#include <neural/json.hpp>
#include <neural/loader.hpp>

typedef std::vector<std::vector<double>> img;


int main() {


    Loader trainer = Loader(LoadType::INT_INT, LoadStyle::DYNAMIC, ModelType::REGRESS, 1, 1, "datasets/IITest.csv");


    Layer* l1 = new InputLayer(1);
    Layer* l2 = new InnerLayer(th, 32);
    Layer* l3 = new InnerLayer(th, 16);
    Layer* l4 = new OutputLayer(nAc,1,16);
    Network nw = Network(4,l1,l2,l3,l4);
    nw.train(trainer, 5000, 16, 0.003);


    Loader tester = Loader(LoadType::INT_INT, LoadStyle::DYNAMIC, ModelType::REGRESS, 1, 1, "datasets/IITest.csv");
    nw.test(tester, 50);

    nw.save("regressNet.spl");    
}






