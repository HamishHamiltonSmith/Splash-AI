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

    Network nw = Network(loadFromFile("mnistNet.spl"));


    Loader tester = Loader(LoadType::PNG_INT, LoadStyle::DYNAMIC, ModelType::CLASSIFY, 28*28, 1*10, "datasets/mnist-pngs/test.csv", "datasets/mnist-pngs/");
    nw.test(tester, 50);
}
