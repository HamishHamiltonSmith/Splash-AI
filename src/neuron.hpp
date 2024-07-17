#pragma once
#include<numeric>


//HAMISH: This is terrible practice please fix
class Neuron;
class Activation {
    public:
    std::function<double(double)> uAc;
    std::function<double(double)> p_uAc;
    std::string name;
    Activation();
    Activation(std::string name);
    Activation(Activation& ac);
    std::string str();
    double ac(double ws);
    double pAc(double ws);
    double ac(std::vector<double>& outputs, int curr);
    std::vector<double> pAc(std::vector<Neuron*>& outputs, int curr);

};


struct PoolData {
    int fm;
    //Bx, By denote position of pool
    int bx;
    int by;
    int idx;
};

double combine(std::vector<double>& i, std::vector<double> w) {
    double total=0;
    
    if (i.size() == w.size()) {
        for (int x=0; x<i.size(); x++) {
            total += i[x] * w[x];
        }
    } else {
        LOG("INVALID WEIGHTED SUM");
        //std::cout << i.size() <<  " | " << w.size();
    }
    return total;
}

class Neuron {
    public:
    Activation activate;
    std::vector<double> weights;
    std::vector<double> regionVals;
    double bias;

    double lastAc;
    double wIn;

    Neuron(Activation ac, std::vector<double> weights, double bias) {
        this->activate = ac;
        this->weights = weights;
        this->bias = bias;
        this->wIn=0;
        this->lastAc=0;
        this->regionVals = std::vector<double>();
    }


    double feed(std::vector<double> in) {
        double z = combine(in, weights)+bias;
        this->wIn = z;
        this->lastAc = activate.ac(z);
        return this->lastAc;
    }
};


