#pragma once
#include <cmath>
#include <map>
#include <functional>
#include <string>

#include "neuron.hpp"



double softmax(const std::vector<double>& outputs, int curr) {
    double sum = 0;
    for (double i: outputs) {
        sum += exp(i);
    }
    return exp(outputs[curr])/sum;
}


double softmax(const std::vector<Neuron*>& outputs, int curr) {
    double sum = 0;
    for (Neuron* n: outputs) {
        sum += exp(n->lastAc);
    }
    return exp(outputs[curr]->lastAc)/sum;
}


std::function<double(double)> TANH = [](double ws) {return tanh(ws);};
std::function<double(double)> SIGMOID = [](double ws) {return 1 / (1 + exp(-ws));};
std::function<double(double)> NAC = [](double ws) {return ws;};
std::function<double(double)> RELU = [](double ws) {return (ws >= 0)? ws : 0.0;};
std::function<double(std::vector<double>&, int)> SOFTMAX = [](std::vector<double>& outputs, int curr) {
    double sum = 0;
    for (double& i: outputs) {
        sum += exp(i);
    }
    return exp(outputs[curr])/sum;
};


std::function<double(double)> P_SIG = [](double ws) {return SIGMOID(ws)*(1-SIGMOID(ws));};
std::function<double(double)> P_RELU = [](double ws) {return (ws >= 0)? 1.0 : 0.0;};
std::function<double(double)> P_TANH = [](double ws) {return pow(1/cosh(ws),2);};
std::function<double(double)> P_NAC = [](double ws) {return 1.0;};
std::function<std::vector<double>(std::vector<Neuron*>&, int)> P_SOFTMAX = [](std::vector<Neuron*>& outputs, int curr) {
    //This function generates a vector of all the derivatives required to perform softmax backprop for a given neuron
    std::vector<double> jVec = std::vector<double>(outputs.size());

    
    int p=0;
    

    for (Neuron* n: outputs) {
        if (curr == p) {
            jVec[p] = softmax(outputs, curr)*(1-softmax(outputs, curr));
        } else {
            jVec[p] = (-softmax(outputs, p)) * softmax(outputs, curr);
        }
        p++; 
    }

    return jVec;
};

std::map<std::string, std::function<double(double)>> acs{{"tanh",TANH}, {"sig",SIGMOID},{"nAc",NAC},{"relu", RELU}};
std::map<std::string, std::function<double(double)>> p_acs{{"tanh",P_TANH},{"sig",P_SIG},{"relu",P_RELU},{"nAc", P_NAC}};


Activation::Activation() {}    
Activation::Activation(std::string name) {
    this->name = name;

    if (acs[name]) {
        this->uAc = acs[name];
        this->p_uAc = p_acs[name];
    } else if (name != "SOFTMAX") {
        std::cout << "\n[+] Unrecognised activation function, exiting...";
        exit(-1);
    } else {
        //Softmax calc performed in layer feedthrough method, neurons should retain weighted sum as activation individual feeds
        this->uAc = NAC;
        this->p_uAc = P_NAC;
    }
}
Activation::Activation(Activation& ac) {
    this->name = ac.str();

    if (acs[name]) {
        this->uAc = acs[name];
        this->p_uAc = p_acs[name];
    } else if (name != "SOFTMAX") {
        std::cout << "\n[+] Unrecognised activation function, exiting...";
        exit(-1);
    } else {
        //Softmax calc performed in layer feedthrough method, neurons should retain weighted sum as activation individual feeds
        this->uAc = NAC;
        this->p_uAc = P_NAC;
    }
}
std::string Activation::str() {
    return this->name; 
}
//Unary overload
double Activation::ac(double ws) {
    return uAc(ws);
}

double Activation::pAc(double ws) {
    return p_uAc(ws);
}

//Softmax overload
double Activation::ac(std::vector<double>& outputs, int curr) {
    return SOFTMAX(outputs,curr);
}

std::vector<double> Activation::pAc(std::vector<Neuron*>& outputs, int curr) {
    return P_SOFTMAX(outputs,curr);
}


Activation smax = Activation("SOFTMAX");
Activation sig = Activation("sig");
Activation th = Activation("tanh");
Activation nAc = Activation("nAc");
Activation relu = Activation("relu");