#pragma once
#include <functional>
#include <algorithm>
#include <deque>
#include <valarray>
#include <random>
#include <iostream>
#include "activations.hpp"
#include "neuron.hpp"




typedef std::vector<std::vector<double>> img;

std::random_device rd;
std::mt19937 eng(rd());
std::normal_distribution<double> norm(0,0.4);


std::ostream& operator<< (std::ostream& os, std::vector<double> v) {
    os << "[";
    for (auto& element: v) {
        os << element << ", ";
    }
    os << "]";
    return os;
}

std::ostream& operator<< (std::ostream& os, std::vector<std::vector<double>> v) {
    os << "[";
    for (auto& element: v) {
        os << element << ", ";
    }
    os << "]";
    return os;
}

class Layer {
    public:
    std::vector<Neuron*> neurons;
    int size;
    Layer* next;
    MATH::Vector<double> errs;
    virtual MATH::Vector<double> backProp(std::vector<double> expected, Layer* prev) = 0;
    virtual void construct(std::deque<Layer*>& layerQ, int c) = 0;
    virtual std::vector<double> feedThrough(std::vector<double>& values) = 0;
    virtual void optimize(int batchSize, double lRate)=0;
    virtual std::string save(std::string& ss)=0;

    int nWeights;
    int nBias;
    
};

class InputLayer : public Layer {
    public:

    std::vector<double> acs;

    std::vector<double> feedThrough(std::vector<double>& values) {
        if (values.size() != this->size) {std::cout << "Input Error: Invalid input length";}
        this->acs = values;
        return next->feedThrough(values);
    }

    InputLayer(int size) {
        this->size = size;
        this->nWeights = size;
        this->nBias = size;
    }

    std::string save(std::string& ss) {
        ss.append("F:" + std::to_string(this->size)+"/");
        this->next->save(ss);
        return ss;
    }

    void construct(std::deque<Layer*>& layerQ, int c) {
        if (layerQ.size() == 0) {throw std::bad_function_call();}

        std::cout << "\n[+] Constructing INPUT Layer: " << c-1 << "\n";

        this->next = layerQ.at(c);
        this->next->construct(layerQ,c+1);
    }

    MATH::Vector<double> backProp(std::vector<double> expected, Layer* prev) {
        return this->next->backProp(expected, this);
    }

    void optimize(int batchSize, double lRate) {
        this->next->optimize(batchSize, lRate);
    }
};



class ConvLayer : public Layer {
    public:


    int regionWidth, regionHeight, nRegions, nFeatureMaps, imgWidth, imgHeight, outputWidth, outputHeight;
    Activation activation;


    std::vector<double> feedThrough(std::vector<double>& values) {
        std::vector<double> output;
        std::vector<double> receptiveField;
        int bx=0;
        int by=0;
        for (int x=0; x<this->size; x+=nFeatureMaps) {

            
            int yChange = by*this->imgWidth;
            for (int y=0; y<this->regionHeight; y++) {
                receptiveField.insert(receptiveField.end(), values.begin()+bx+yChange, values.begin()+bx+yChange+this->regionWidth);
                yChange+=this->imgWidth;
            }


            bx++;
            if (bx+this->regionWidth>this->imgWidth) {
                bx=0;
                by++;
            }


            for (int y=0; y<this->nFeatureMaps; y++) {
                double val = this->neurons.at(x+y)->feed(receptiveField);

                //Receptive field capture for backprop
                this->neurons.at(x+y)->regionVals = receptiveField;
                output.push_back(val);
            }
            receptiveField.clear();
        }

        
        return next->feedThrough(output);
    }

    ConvLayer(Activation ac, int nFeatureMaps, int regionWidth, int regionHeight, int imgWidth, int imgHeight, std::vector<double> loadBias = std::vector<double>(), std::vector<std::vector<double>> loadWeight = std::vector<std::vector<double>>(), bool load=false) {
        int overX = regionWidth-1;
        int overY = regionHeight-1;
        this->activation = ac;

        this->outputWidth = imgWidth-overX;
        this->outputHeight = imgHeight-overY;
        this->nRegions = this->outputHeight*this->outputWidth;
         
        this->nFeatureMaps = nFeatureMaps;
        this->regionWidth = regionWidth;
        this->regionHeight = regionHeight;
        this->size = nRegions*nFeatureMaps;
        this->imgWidth = imgWidth;
        this->imgHeight = imgHeight;
        this->nBias = size;
        this->nWeights = (regionWidth*regionHeight)*size;

        std::vector<std::vector<double>> sharedWeights(nFeatureMaps,std::vector<double>());


        //Sharing of weights & bias in feature maps
        if (!load) {
            for (int fm=0; fm<this->nFeatureMaps; fm++) {

                for (int y=0; y<(regionWidth*regionHeight); y++) {
                    sharedWeights.at(fm).push_back((norm(eng)));
                }
            }

            for (int x=0; x<size; x++) {
                neurons.push_back(new Neuron(activation , sharedWeights[x%nFeatureMaps], 0));
            }

        } else {
            sharedWeights = loadWeight;
            for (int x=0; x<size; x++) {
                neurons.push_back(new Neuron(activation , sharedWeights[x%nFeatureMaps], loadBias[x]));
            }
        }



    }

    std::string save(std::string& ss) {

        std::string weights;
        for (int x=0; x<this->nFeatureMaps; x++) {
            std::for_each(this->neurons.at(x)->weights.begin(), this->neurons.at(x)->weights.end(), [&](double& f){weights.append(std::to_string(f)+",");});
        }
        
        std::string bias;
        for (int x=0; x<this->size; x++) {
            bias.append(std::to_string(this->neurons[x]->bias) + ",");
        }

        ss.append("C:"+this->activation.str()+":"+std::to_string(this->nFeatureMaps)+":"+std::to_string(this->regionWidth)+":"+std::to_string(this->regionHeight)+":"+std::to_string(this->imgWidth)+":"+std::to_string(this->size)+":"+weights+":"+bias+"/");
        this->next->save(ss);
        return ss;
    }

    void construct(std::deque<Layer*>& layerQ, int c) {
        if (layerQ.size() == 0) {throw std::bad_function_call();}

        std::cout << "[+] Constructing CONV Layer: " << c-1 << "\n";
        
        this->next = layerQ.at(c);
        this->next->construct(layerQ,c+1);
    }

    ~ConvLayer() {
        delete next;
        for (auto p: neurons) {
            delete p;
        }
        neurons.clear();
    } 

    MATH::Vector<double> backProp(std::vector<double> expected, Layer* prev) {
        return next->backProp(expected, this);
    }

    void optimize(int batchSize, double lRate) {
        this->next->optimize(batchSize,lRate);
    }
    
};


class MPoolLayer : public Layer {
    public:
    
    //Number of fmaps
    int prevReceptiveFields;
    std::vector<PoolData> pools;

    //Width&height of conv image for each fmap
    int receptiveHeight;
    int receptiveWidth;
    MATH::Vector<double> weightErrs;
    double biasErr;
    ConvLayer* prev;
    Activation prevActivation;


    //eg: 2x2
    int poolSize;

    std::vector<double> feedThrough(std::vector<double>& values) {
        std::vector<double> output;
        std::vector<std::vector<double>> fMaps(prevReceptiveFields,std::vector<double>());
        int prevSize = values.size();
        pools.clear();

        //Init feature maps with values
        for (int x=0; x<values.size(); x++) {
            fMaps[x%this->prevReceptiveFields].push_back(values[x]);
        }

        //Max pool feature maps
        int f = 0;
        int count = 0;
        for (auto& fm: fMaps) {
            std::vector<double> poolQuad;
            int bx=0;
            int by=0;
            int yChange = 0;
            
            while (by+this->poolSize <= this->receptiveHeight) {

                yChange = by*receptiveWidth;

                for (int y=0; y<this->poolSize; y++) {
                    //load pools with 2x2 samples
                    poolQuad.insert(poolQuad.end(), fm.begin()+bx+yChange, fm.begin()+bx+yChange+this->poolSize);
                    yChange += receptiveWidth;
                }
             


                auto mPointer = std::max_element(poolQuad.begin(),poolQuad.end());
                double max = *mPointer;

                int dist = (int)std::distance(poolQuad.begin(), mPointer);
                if (dist > 1) {
                    //Account for lines
                    dist += this->receptiveWidth-this->poolSize;
                }
   
                pools.push_back({f,bx,by*receptiveWidth, dist});
                output.push_back(max);
                this->neurons.at(count)->lastAc = max;

                poolQuad.clear();
                count++;

                bx+=poolSize;
                if (bx+this->poolSize > receptiveWidth) {
                    bx = 0;
                    by+=poolSize;
                }       
            }
            f++;
        }

        return next->feedThrough(output);
    }

    MPoolLayer(int poolSize) {
        this->poolSize = poolSize;
    }

    std::string save(std::string& ss) {
        ss.append("M:" + std::to_string(this->poolSize) + "/");
        this->next->save(ss);
        return ss;
    }

    void construct(std::deque<Layer*>& layerQ, int c) {
        if (layerQ.size() == 0) {throw std::bad_function_call();}
        this->next = layerQ.at(c);

        std::cout << "[+] Constructing MPOOL Layer: " << c-1 << "\n";

        this->prev = dynamic_cast<ConvLayer*>(layerQ.at(c-2));
        if (prev == nullptr) {std::cout<<"\nMPool MUST proceed Conv layer\n";throw std::bad_function_call();}
        this->size = prev->nFeatureMaps * ((floor(prev->outputWidth/this->poolSize))*floor(prev->outputHeight/this->poolSize));
        for (int x=0; x<this->size; x++) {
            this->neurons.push_back(new Neuron(nAc,std::vector<double>(),0));
        }
        this->prevReceptiveFields = prev->nFeatureMaps;
        this->receptiveWidth = prev->outputWidth;
        this->receptiveHeight = prev->outputHeight;
        this->prevActivation = prev->activation;
        weightErrs = MATH::Vector<double>(prev->regionWidth*prev->regionHeight*prev->nFeatureMaps,0);
        //biasErrs = MATH::Vector<double>(this->size,0);
        this->next->construct(layerQ, c+1);
    }

    MATH::Vector<double> backProp(std::vector<double> expected, Layer* previous) {
        MATH::Vector<double> forwardErr = this->next->backProp(expected, this);

        

        //FM - Denotes feature map bias for the weight errors
        int f = 0;

        //COUNT, poolIdx: Counts each pool instance for fm calculation
        int poolIdx = 0;
        double lRate = 0.03;

        //A vector of weight errors - appropriate sections updated for each convolutional neuron.
        //Based on the neurons feature map - a diferent section should be updated.
        for (PoolData data: this->pools) {

            //Access MAX activation neuron
            f = data.fm*this->prev->regionWidth*this->prev->regionHeight;
            int winIdx = (data.bx + data.by + data.idx) * (this->prev->nFeatureMaps) + data.fm;
            Neuron* winner = this->prev->neurons.at(winIdx);

            //Loop through every weight the neuron has
            for (int x=0; x<winner->weights.size(); x++) {
                //Access coresponging region value

                int idx=0; 
                for (Neuron* n : this->next->neurons) {
                    weightErrs.at(x+f) += this->prevActivation.pAc(winner->wIn)*forwardErr.at(idx)*n->weights.at(poolIdx)*winner->regionVals.at(x);
                    idx++; 
                }

            }

            int idx=0;
            this->biasErr = 0;
            for (Neuron* n : this->next->neurons) {
                this->biasErr += (forwardErr.at(idx)*n->weights.at(poolIdx));
                idx++;
            }
            //HAMISH ---------------------------------------------- PROBLEM BIAS SET AT 100 BARELY CHANGES BIAS LEARNING ALSO WHY NO PROPER SAVE FIX
            winner->bias -= this->biasErr*lRate;

            
            poolIdx++;
        }

        return forwardErr;
    }

    void optimize(int batchSize, double lRate) {
        weightErrs = this->weightErrs*(1/(double)batchSize);

        
        int fm=0;
        int bias=0;

        //Update feature maps for each neuron
        for (int x=0; x<this->prev->neurons.size(); x++) {
            fm = x%this->prev->nFeatureMaps;

            bias = fm*this->prev->regionHeight*this->prev->regionWidth;
            
            Neuron* n = this->prev->neurons.at(x);
            for (int y=0; y<n->weights.size(); y++) {
                n->weights.at(y) -= (weightErrs.at(bias+y) * lRate);
            }
        }   


        this->next->optimize(batchSize, lRate);
        this->weightErrs.zero();
    }
};


class InnerLayer : public Layer {
    public:
    int lastSize = 0;
    int iWeight;
    int iBias;
    
    bool isPreset;
    bool isLoad;
    Activation activation; 

    MATH::Vector<double> weightDeriv;
    MATH::Vector<double> biasDeriv;
    std::vector<double> weightLoad;
    std::vector<double> biasLoad;

    std::vector<double> feedThrough(std::vector<double>& values) {
        std::vector<double> output;
        for (int x=0; x<this->size; x++) {
            output.push_back(neurons.at(x)->feed(values));
        }
        return next->feedThrough(output);
    }

    InnerLayer(Activation ac, int size, std::vector<double> biasLoad=std::vector<double>(), std::vector<double> weightLoad=std::vector<double>(), bool isLoad=false) {
        this->isPreset = false;
        this->isLoad = isLoad;
        this->size = size;
        this->activation = ac;
        this->lastSize = 0;
        this->nBias = size;
        this->weightLoad = weightLoad;
        this->biasLoad = biasLoad;
    }

    InnerLayer(Activation ac, int size, double weight, double bias=0) {
        this->isPreset = true;

        this->lastSize = 0;
        this->nWeights = 0;
        this->iWeight = weight;
        this->activation = ac;
        this->iBias = bias;
        this->nBias = size;        
        this->isLoad = false;
        this->weightLoad = std::vector<double>();
        this->biasLoad = std::vector<double>();
        this->size = size;
    }

    std::string save(std::string& ss) {
        std::string weights;
        std::string bias;

        std::for_each(this->neurons.begin(), this->neurons.end(), [&](Neuron* n){
            bias.append(std::to_string(n->bias) + ",");
            std::for_each(n->weights.begin(), n->weights.end(),[&](double& w){weights.append(std::to_string(w)+",");});

            //std::cout << "Len: " << n->weights.size();
        });
       

        
        ss.append("I:"+this->activation.str()+":"+std::to_string(this->size)+":"+bias+":"+weights+"/");
        this->next->save(ss);
        return ss;
    }

    void construct(std::deque<Layer*>& layerQ, int c) {
        if (layerQ.size() == 0) {throw std::bad_function_call();}

        std::cout << "[+] Constructing INNER Layer: " << c-1 << "\n";
        
        this->next = layerQ.at(c);
        this->lastSize = layerQ.at(c-2)->size;
        this->nWeights = size*lastSize;

        this->weightDeriv = MATH::Vector<double>(nWeights,0);
        this->biasDeriv = MATH::Vector<double>(nBias,0);

        if (isPreset) {
            std::vector<double> weights(lastSize, iWeight);
            for (int x=0; x<size; x++) {   
                neurons.push_back(new Neuron(activation, weights, iBias));
            }
        } else if (isLoad) {
            std::vector<double> weights = std::vector<double>{};
            int c=0;
            for (int x=0; x<size; x++) {
                for (int y=0; y<this->lastSize; y++) {
                    weights.push_back(this->weightLoad[c]);
                    c++;
                }

                neurons.push_back(new Neuron(activation , weights, this->biasLoad[x]));
                weights.clear();
            }
        } else {
            std::vector<double> weights = std::vector<double>{};
            for (int x=0; x<size; x++) {


                for (int y=0; y<this->lastSize; y++) {
                    weights.push_back(norm(eng));
                }

                neurons.push_back(new Neuron(activation , weights, 0));
                weights.clear();
            }
        }

        this->next->construct(layerQ,c+1);
    }

    ~InnerLayer() {
        delete next;
        for (auto p: neurons) {
            delete p;
        }
        neurons.clear();
    } 

    MATH::Vector<double> backProp(std::vector<double> expected, Layer* prev) {
        MATH::Vector<double> forwardErr = next->backProp(expected, this);

        this->errs.clear();
        int bias = 0;
        double totalErr=0;
    


        for (int x=0; x<this->neurons.size(); x++) {
            Neuron* n = this->neurons.at(x);
            totalErr=0;

            bias = x*n->weights.size();
            
            //Calculate error
            for (int y=0; y<this->next->neurons.size(); y++ ){

                // --    
                totalErr += this->activation.pAc(n->wIn)*this->next->neurons.at(y)->weights.at(x)*forwardErr.at(y);
                // --
            }

            this->errs.push_back(totalErr);

            //Compute weight and bias gradients for input 

            if (prev->neurons.size() != 0) {
                for (int z=0; z<n->weights.size(); z++) {
                    weightDeriv.at(bias+z) += totalErr*prev->neurons.at(z)->lastAc;
                }
            } else {
                for (int z=0; z<n->weights.size(); z++) {
                    weightDeriv.at(bias+z) += totalErr*dynamic_cast<InputLayer*>(prev)->acs.at(z);
                }
            }
            
            biasDeriv.at(x) += totalErr;
        }
        
        return this->errs;
    }

    void optimize(int batchSize, double lRate) {
        int bias=0;

        //Calculate Weight 
        weightDeriv = weightDeriv*(1/(double)batchSize);
        biasDeriv = biasDeriv*(1/(double)batchSize);

        for (int y=0; y<this->neurons.size(); y++) {
            Neuron* n = this->neurons.at(y);
            bias = y*n->weights.size();


            for (int x=0; x<n->weights.size(); x++) {
                n->weights.at(x) -= lRate*weightDeriv.at(bias+x);
            }
            n->bias -= (lRate)*biasDeriv.at(y);
        }

        this->next->optimize(batchSize, lRate);

        this->weightDeriv.zero();
        this->biasDeriv.zero();
    }
    
};

class OutputLayer : public Layer {
    public:

    MATH::Vector<double> weightDeriv;
    MATH::Vector<double> biasDeriv;
    std::vector<double> weightLoad;
    std::vector<double> biasLoad;
    Activation activation;
    bool isLoad;

    std::vector<double> feedThrough(std::vector<double>& values) {
        std::vector<double> output;

        for (int x=0; x<this->size; x++) {
            output.push_back(neurons.at(x)->feed(values));
        }
        
        if (this->activation.str() == "SOFTMAX") {
            for (int n=0; n<this->size; n++) {
                output[n] = this->activation.ac(output, n);
            }
        }

        return output;
    }


    OutputLayer(Activation ac, int size, int last, std::vector<double> biasLoad = std::vector<double>(), std::vector<double> weightLoad = std::vector<double>(), bool isLoad = false) {
        this->nWeights = size*last;
        this->nBias = size;
        this->weightDeriv = MATH::Vector<double>(nWeights,0);
        this->biasDeriv = MATH::Vector<double>(nBias,0);
        this->activation = ac;
        this->isLoad = isLoad;
        this->weightLoad = weightLoad;
        this->biasLoad = biasLoad;
        this->errs = std::vector<double>();

        if (isLoad) {
            std::vector<double> weights;
            int c=0;

            for (int x=0; x<size; x++) {
                for (int y=0; y<last; y++) {
                    weights.push_back(weightLoad[c]);
                    c++;
                }
                neurons.push_back(new Neuron(activation , weights, this->biasLoad[x]));
                weights.clear();
            }
        } else {
            std::vector<double> weights;
            for (int x=0; x<size; x++) {
                for (int y=0; y<last; y++) {

                    weights.push_back(norm(eng));
                }
                neurons.push_back(new Neuron(activation , weights, 0));
                weights.clear();
            }
        }

        this->size = size; 
    }

    OutputLayer(Activation ac, int size, int last, double weight, double bias=0) {
        this->nWeights = size*last;
        this->nBias = size;
        weightDeriv = MATH::Vector<double>(nWeights,0);
        biasDeriv = MATH::Vector<double>(nBias,0);
        this->activation = ac;
        this->isLoad = false;
        this->weightLoad = std::vector<double>();
        this->biasLoad = std::vector<double>();
        this->errs = std::vector<double>();


        std::vector<double> weights(last, weight);
        for (int x=0; x<size; x++) {
            neurons.push_back(new Neuron(activation, weights, bias));
        }

        this->size = size;
    }

    std::string save(std::string& ss) {
        std::string weights;
        std::string bias;
        std::for_each(this->neurons.begin(), this->neurons.end(), [&](Neuron* n){
            std::for_each(n->weights.begin(), n->weights.end(),[&](double& w){weights.append(std::to_string(w)+",");});
            bias.append(std::to_string(n->bias) + ",");
        });

        ss.append("O:"+this->activation.str()+":"+std::to_string(this->size)+":"+weights+":"+bias+"/");

        return ss;
    }

    void construct(std::deque<Layer*>& LayerQ, int c) {
        std::cout << "[+] Constructing OUTPUT Layer: " << c-1 << "\n\n";
        this->next = nullptr;
    }

    ~OutputLayer() {
        delete next;
        for (Neuron* p: this->neurons) {
            delete p;
        }
        neurons.clear();
    }

    MATH::Vector<double> backProp(std::vector<double> expected, Layer* prev) {
        this->errs.clear();
        int bias=0;
        double ex = 0;
        double err = 0;


        if (this->activation.str() == "SOFTMAX") {
            MATH::Vector<double> derivs = MATH::Vector<double>(this->neurons.size(), 0);
            errs = MATH::Vector<double>(this->neurons.size(), 0);

            for (int x=0; x<this->neurons.size(); x++) {
                derivs = this->activation.pAc(this->neurons, x);
                for (int y=0; y<this->neurons.size(); y++) {
                    Neuron* n = neurons.at(y);
                    ex = expected.at(y);

                    this->errs.at(x) += derivs.at(y)*(n->lastAc - ex);
                }

                for (int z=0; z<this->neurons[x]->weights.size(); z++) {
                    weightDeriv.at(bias+z) += this->errs.at(x)*prev->neurons.at(z)->lastAc;
                }

                biasDeriv.at(x) += this->errs.at(x);
            }


        } else {
            for (int x=0; x<this->neurons.size(); x++) {
                //Calculate activation error for output
                Neuron* n = neurons.at(x);
                err = 0;
                ex = expected.at(x);

                err = this->activation.pAc(n->wIn)*(n->lastAc - ex);
                errs.push_back(err);

                bias = x*n->weights.size();

                //Compute weight and bias gradients for input 
                for (int z=0; z<n->weights.size(); z++) {
                    weightDeriv.at(bias+z) += err*prev->neurons.at(z)->lastAc;
                }

                
                biasDeriv.at(x) += err;
            }
        }

        return this->errs;
    } 

    void optimize(int batchSize, double lRate) {
        int bias=0;
        weightDeriv = weightDeriv*(1/(double)batchSize);
        biasDeriv = biasDeriv*(1/(double)batchSize);


        for (int y=0; y<this->neurons.size(); y++) {
            Neuron* n = neurons.at(y);
            bias  = y*n->weights.size();

            for (int x=0; x<n->weights.size(); x++) {
                n->weights.at(x) -= (lRate)*this->weightDeriv.at(bias+x);
            }
            n->bias -= (lRate)*biasDeriv.at(y);
        }

        this->weightDeriv.zero();
        this->biasDeriv.zero();
    }
};
