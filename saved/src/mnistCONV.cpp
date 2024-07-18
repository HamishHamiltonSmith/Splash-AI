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
    Layer* l2 = new ConvLayer(sig,2,3,3,28,28);
    Layer* l3 = new MPoolLayer(2);
    Layer* l4 = new InnerLayer(sig, 100);
    Layer* l5 = new OutputLayer(sig,10,100);
    Network nw = Network(5,l1,l2,l3,l4,l5);
    nw.train(trainer, 1000, 3, 0.06);

    Loader tester = Loader(LoadType::PNG_INT, LoadStyle::DYNAMIC, ModelType::CLASSIFY, 28*28, 1*10, "datasets/mnist-pngs/test.csv", "datasets/mnist-pngs/");
    nw.test(tester, 1000); 

    //int i;
    //std::cin >> i;
    //std::vector<double> out;
    //std::vector<double> in;
    //while (i != -1) {
    //    in = std::get<0>(tester.pull(i));
    //    out = nw.feedThrough(in);
    //    std::cout << "\nExpected: " << std::get<1>(tester.pull(i)) << std::endl;
    //    std::cout << "Actual: " << out << std::endl;
    //    std::cin >> i;
    //}
    
    nw.save("mnistNetCONV.spl");
}





