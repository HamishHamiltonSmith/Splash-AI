//Debug
#define debug

#ifdef debug 
#define LOG(msg) std::cout<<"LOG: " << msg << std::endl
#else
#define LOG(msg)
#endif


//Standard library
#include <algorithm>
#include <vector>
#include <fstream>
#include <math.h>
#include <string>
#include <iostream>
#include <thread>
#include <sstream>
#include <functional>

//Splash Modules
#include <neural/math.hpp>
#include <neural/filehandle.hpp>
#include <neural/network.hpp>
#include <neural/activations.hpp>
#include <neural/loader.hpp>


int main() {

    Loader trainer = Loader(LoadType::PNG_INT, LoadStyle::DYNAMIC, ModelType::CLASSIFY, 28*28, 1*10, "datasets/mnist-pngs/train.csv", "datasets/mnist-pngs/");


    Layer* l1 = new InputLayer(28*28);
    Layer* l2 = new InnerLayer(th, 100);
    Layer* l3 = new InnerLayer(th, 50);
    Layer* l4 = new OutputLayer(sig,10,50);
    Network nw = Network(4,l1,l2,l3,l4);
    nw.train(trainer, 5000, 5, 0.03);
    
    nw.save("mnistNet.spl");
}





