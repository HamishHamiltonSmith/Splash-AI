#pragma once
#include <cstdarg>
#include <deque>
#include <string>
#include <random>
#include <tuple>
#include <memory>
#include <iostream>

#include "loader.hpp"
#include "layer.hpp"

#define LR 0.003

typedef std::vector<std::vector<double>> img;

int Lm(std::string d) {
    return d.find(":");
}


class Network {
    public:

    Layer* first;
    double lRate;
    int nEpochs;
    int currEpoch;

    std::vector<double> feedThrough(std::vector<double>& input) {
        return first->feedThrough(input);
    }

    Network(std::string tokens) {
        std::string data;
        auto i = tokens.find("/");

        std::deque<Layer*> layerQ;

        int imgSize = 0;
        int imgWidth = 0;
        int imgHeight = 0;
        int mWidth = 0;
        int nMaps = 0;
        int size = 0;
        int last = 0;
        int mHeight = 0;
        int comma = 0;
        std::vector<double> weights;
        std::vector<double> bias;
        std::string activation;
        std::vector<std::vector<double>> sharedWeights;

        while (i != std::string::npos) {
            data = tokens.substr(0,i+1);
            tokens.erase(0,(int)i+1);
            i = tokens.find("/");


            switch (data.c_str()[0]) {
                case 'F':
                    //Input layer
                    data.erase(0,2);
                    imgSize = std::stoi(data.substr(0,Lm(data)));
                    layerQ.push_back(new InputLayer(std::stoi(data.substr(0,Lm(data)))));
                    
                    break; 
                case 'C':
                    //Conv layer
                    data.erase(0,2);
                    activation = data.substr(0,Lm(data));
                    data.erase(0,Lm(data)+1);
                    nMaps = std::stoi(data.substr(0, Lm(data)));
                    data.erase(0,Lm(data)+1);
                    mWidth = std::stoi(data.substr(0, Lm(data)));
                    data.erase(0,Lm(data)+1);
                    mHeight = std::stoi(data.substr(0, Lm(data)));
                    data.erase(0,Lm(data)+1);
                    imgWidth = std::stoi(data.substr(0, Lm(data)));
                    data.erase(0,Lm(data)+1);
                    size = std::stoi(data.substr(0, Lm(data)));
                    data.erase(0,Lm(data)+1);

                    sharedWeights = std::vector<std::vector<double>>(nMaps,std::vector<double>());
                    bias = std::vector<double>();
                    
                

                    for (int x=0; x<nMaps; x++) {
                        for (int y=0; y<mWidth*mHeight; y++) {
                            comma = data.find(",");
                            sharedWeights[x].push_back(std::stod(data.substr(0,comma)));
                            data.erase(0,comma+1);
                        }
                    }

                    data.erase(0, Lm(data)+1);

                    for (int x=0; x<size; x++) {
                        comma = data.find(",");
                        bias.push_back(std::stod(data.substr(0,comma)));
                        data.erase(0, comma+1);
                    }

                    
                    layerQ.push_back(new ConvLayer(activation, nMaps, mWidth, mHeight, imgWidth, imgSize/imgWidth, bias, sharedWeights, true));
                    break; 

                case 'M':
                    data.erase(0,2);
                    layerQ.push_back(new MPoolLayer(std::stoi(data.substr(0,data.find("/")))));
                    break;

                case 'I':
                    data.erase(0,2);
                    activation = data.substr(0,Lm(data));
                    data.erase(0,Lm(data)+1);
                    size = std::stoi(data.substr(0, Lm(data)));
                    data.erase(0,Lm(data)+1);


                    weights = std::vector<double>();
                    bias = std::vector<double>();

                    for (int x=0; x<size; x++) {
                        comma = data.find(",");
                        bias.push_back(std::stod(data.substr(0,comma)));
                        data.erase(0,comma+1);
                    }

                    data.erase(0, data.find(":")+1);
                

                    while (data.find("/")>0) {
                        comma = data.find(",");
                        weights.push_back(std::stod(data.substr(0,comma)));
                        data.erase(0,comma+1);
                    }


                    
                    layerQ.push_back(new InnerLayer(activation, size, bias, weights, true));
                    break; 

                case 'O':
                    data.erase(0,2);
                    activation = data.substr(0,Lm(data));
                    data.erase(0,Lm(data)+1);
                    last = size;
                    size = std::stoi(data.substr(0, Lm(data)));
                    data.erase(0,Lm(data)+1);


                    weights = std::vector<double>();
                    bias = std::vector<double>();                    
                    
                    for (int y=0; y<size; y++) {
                        for (int x=0; x<last; x++) {
                            comma = data.find(",");
                            weights.push_back(std::stod(data.substr(0,comma)));
                            data.erase(0,comma+1);
                        }
                    }

                    data.erase(0, data.find(":")+1);

                    for (int x=0; x<size; x++) {
                        comma = data.find(",");
                        bias.push_back(std::stod(data.substr(0,comma)));
                        data.erase(0,comma+1);
                    }

                    
                    layerQ.push_back(new OutputLayer(activation, size, last, bias, weights, true));
                    break;
                
                default:
                    std::cout << "ERR: Invalid load sequence";
            }
        }

        //Construct network
        this->lRate = LR;
        this->first = layerQ.at(0);
        this->first->construct(layerQ,1);
    }

    Network(int n, ...) {
        va_list layers;
        va_start(layers, n);
        std::deque<Layer*> layerQ;
        Layer* prev;
        this->lRate = LR;

        for (int x=0; x<n; x++) {
            Layer* l = va_arg(layers, Layer*);
            layerQ.push_back(l);
        }

        //Construct network
        this->first = layerQ.at(0);
        this->first->construct(layerQ,1);
    }

    double decay() {
        //return this->lRate*(1/((double)this->currEpoch*0.00005+1));
        return this->lRate;
    }

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& expected, int epochs, int batchSize, double lr = LR) {
        std::vector<std::tuple<std::vector<double>, std::vector<double>>> batch;
        std::default_random_engine gen(time(0));
        std::uniform_int_distribution<int> randInput(0, inputs.size()-1);

        this->nEpochs = epochs;
        this->lRate = lr;
        this->currEpoch = 0;
  
        std::cout << "\n[+] Begiinning static training... This could take a few minutes\n";

        for (int x=0; x<epochs; x++) {
            for (int y=0; y<batchSize; y++) {
                int choice = randInput(gen);

                batch.push_back(std::make_tuple(inputs.at(choice), expected.at(choice))); 

            }

            trainBatch(batch);
            batch.clear();
        }
    }

    void train(const Loader& loader, int epochs, int batchSize, double lr = LR) {
        std::vector<std::tuple<std::vector<double>, std::vector<double>>> batch;
        std::default_random_engine gen(time(0));
        std::uniform_int_distribution<int> randInput(0, loader.fileSize-1);

        this->nEpochs = epochs;
        this->lRate = lr;
        this->currEpoch = 0;
  
        std::cout << "\n[+] Beginning dynamic training... This could take a few minutes\n";

        for (int x=0; x<epochs; x++) {
            for (int y=0; y<batchSize; y++) {
                int choice = randInput(gen);

                batch.push_back(loader.pull(choice)); 

            }

            trainBatch(batch);
            batch.clear();
        }
    }


    double trainBatch(std::vector<std::tuple<std::vector<double>, std::vector<double>>> batch) {
        double cost = 0;

        for (auto& io: batch) {
            std::vector<double> flat = std::get<0>(io);
            std::vector<double> out = this->feedThrough(flat);

            //std::cout  << "\nOUT: " << out;
            

            double loss=0;

            MATH::Vector<double> ex = MATH::Vector<double>(std::get<1>(io));
            //std::cout << "\nEX: " << ex.core;

            for (int x=0; x<out.size(); x++) {
                loss += pow((out.at(x)-ex.at(x)),2);
            }

            cost += loss;
            //std::cout << "\nLOSS: " << loss << std::endl;

            this->first->backProp(std::get<1>(io), nullptr);
        }
        cost *= (1/(double)batch.size());
        std::cout << "\n---COST: " << cost << " ---\n";
        this->first->optimize(batch.size(), this->decay());

        return cost;
    } 


    void test(const Loader& loader, int iterations) {
        std::default_random_engine gen(time(0));
        std::uniform_int_distribution<int> randInput(0, loader.fileSize-1);

        double accuracy = 0;
        std::vector<double> out;
        int maxIdx = 0;
        int ans = 0;

        
        if (loader.ls == LoadStyle::DYNAMIC) {
            if (loader.mt == ModelType::REGRESS) {
                for (int x=0; x<iterations; x++) {

                    std::tuple<std::vector<double>, std::vector<double>> batchPair = loader.pull(randInput(gen));
                    out = this->feedThrough(std::get<0>(batchPair));

                    int c=0;
                    for (auto n: out) {
                        accuracy += pow(( std::get<1>(batchPair)[c] - n), 2);
                        c++;
                    }
                }

                std::cout << "[+] Model loss: " << accuracy/iterations << std::endl;

            } else if (loader.mt == ModelType::CLASSIFY || loader.mt == ModelType::BINARY) {
                for (int x=0; x<iterations ; x++) {

                    std::tuple<std::vector<double>, std::vector<double>> batchPair = loader.pull(randInput(gen));
                    out = this->feedThrough(std::get<0>(batchPair));
                    maxIdx = std::max_element(out.begin(), out.end()) - out.begin();
                    ans = std::max_element(std::get<1>(batchPair).begin(), std::get<1>(batchPair).end()) - std::get<1>(batchPair).begin();

                    if (maxIdx == ans) {
                        accuracy += 1;
                    }
                    
                }

                std::cout << "[+] Model accuracy: " << (accuracy/iterations)*100 << "%\n";                
            }

        } else {
            std::cout << "\n[+] Invalid loader, set loader style to DYNAMIC for testing\n";
        }
    }


    void structure() {
        Layer* next = first->next;
        int c = 1;
        while (next != nullptr) {
            if (c == 3) {continue;}
            std::cout << "\nLAYER: " <<  c << "\n";  
            for (auto n: next->neurons) {
                for (auto w: n->weights) {
                    std::cout << "[" << std::to_string(w) << "]";
                }
                std::cout << "\n";
            }
            next = next->next;
            c++;
        }
    }

    void save(std::string fname) {
        //FORMAT: 
        //Cw1:b1:nFM:3:3/Pw2:b2:2:2/Iw3:b3/Ow4:b4
        std::string str;
        this->first->save(str);
        saveToFile(fname,str);
    }


    void state() {
        Layer* next = first->next;
        int lc = 1;
        while (next != nullptr) {
            std::cout << "\n\nLayer " << lc << std::endl;
            for (auto n: next->neurons) {
                std::cout << "\n\x1B[31mNeuron\033[0m\t\t" << std::endl << "Weights:";
                for (auto w: n->weights) {
                    std::cout << "W: " << w << "  ";
                }
                std::cout << "\nBias: " << n->bias;

                std::cout << "\nWeighted Input " << n->wIn;
                std::cout << "\nLast activation: " << n->lastAc;
            }
            next = next->next;
            lc++;
        }
    }

    ~Network() {
        delete first;
    }
};
