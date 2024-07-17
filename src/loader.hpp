#pragma once

//The loader can be used for the easy full loading of datasets stored in a
//suitably formatted text or csv file. This can be memory intensive. 
//For larger datasets I recommend you partially load. 


//Expected file format:
// value, value

#include "filehandle.hpp"
#include <filesystem>


struct TrainingData {
    std::vector<std::vector<double>> xTrain;
    std::vector<std::vector<double>> yTrain;
};


namespace fs = std::filesystem;

//TODO: Implement VEC_VEC 


enum class LoadStyle {DYNAMIC, STATIC};
enum class LoadType {PNG_INT, INT_PNG, INT_INT, VEC_VEC};
enum class ModelType {CLASSIFY, BINARY, REGRESS};

std::function<std::tuple<std::vector<double>, std::vector<double>>(std::string, LoadType, ModelType, int, const std::string&)> pull_int_int = [](std::string line, LoadType lt, ModelType mt, int outSize, const std::string& pathInsert){
    return std::make_tuple(std::vector<double>{std::stod(line.substr(0,line.find(",")))}, std::vector<double>{std::stod(line.substr(line.find(",")+1, line.size()))});
};

std::function<std::tuple<std::vector<double>, std::vector<double>>(std::string, LoadType, ModelType, int, const std::string&)> pull_png_int = [](std::string line, LoadType lt, ModelType mt, int outSize, const std::string& pathInsert){
    //Check modeltype here
    double score;
    std::vector<double> xTrain;
    
    (lt==LoadType::PNG_INT) ? score = std::stod(line.substr(line .find(",")+1, line.size())) : score = std::stod(line.substr(0,line.find(",")));
    (lt==LoadType::PNG_INT) ? xTrain = MATH::flatten(getPng(pathInsert + line.substr(0,line.find(",")))): xTrain = MATH::flatten(getPng(pathInsert + line.substr(line.find(",")+1,line.size())));
    

    if (mt == ModelType::CLASSIFY || mt == ModelType::BINARY) {
        std::vector<double> exVec = std::vector<double>(outSize,0);
        exVec[(int)score] = 1;

        return std::make_tuple(xTrain, exVec);
    } else {
        return std::make_tuple(xTrain, std::vector<double>{score});
    }
};






struct Loader {

    std::function<std::tuple<std::vector<double>, std::vector<double>>(std::string, LoadType, ModelType, int, const std::string&)> pullFunc;
    TrainingData trainingPair;
    std::ifstream* fp;
    ModelType mt;
    LoadType lt;
    LoadStyle ls;
    double inputSize;
    double outputSize;
    int fileSize;
    std::string pathInsert;

    //File Init
    Loader(LoadType lt, LoadStyle ls, ModelType mt, int inputSize, int outputSize, std::string fname, std::string pathInsert="", int cutOff = 0) {

        this->fp = new std::ifstream(fname);
        this->fileSize = std::count_if(std::istreambuf_iterator<char>{ (*this->fp) }, {}, [](char c) { return c == '\n'; });
        this->fp->clear();
        this->fp->seekg(0);

        this->mt = mt;
        this->lt = lt;
        this->ls = ls;
        this->inputSize = inputSize;
        this->outputSize = outputSize;
        this->pathInsert = pathInsert;

        //Vector for output 
        std::string line;
        int n=0;
        std::vector<double> inVec;
        std::vector<double> exVec;


        switch (lt) {
            case LoadType::INT_INT:
                this->pullFunc = pull_int_int;

                if (ls == LoadStyle::STATIC) {

                    while (std::getline(*this->fp , (line) )) {
                        std::cout << "\r[+] Loading Line " << n+1 << std::flush;

                        trainingPair.xTrain.push_back( std::vector<double>{std::stod(line.substr(0,line.find(",")))} );
                        trainingPair.yTrain.push_back( std::vector<double>{std::stod(line.substr(line.find(",")+1, line.size()))} );

                        if (cutOff && n > cutOff) {
                            break;
                        } 

                        n++;
                    }

                    std::cout << std::endl;

                }

                break;

            case LoadType::PNG_INT: 
            case LoadType::INT_PNG:
                this->pullFunc = pull_png_int;


                if (ls == LoadStyle::STATIC) {
                    double score;

                    while (std::getline(*this->fp , (line) )) {
                        std::cout << "\r[+] Loading file " << n+1 << std::flush;

                        
                        (lt==LoadType::PNG_INT) ? score = std::stod(line.substr(line .find(",")+1, line.size())) : score = std::stod(line.substr(0,line.find(",")));
                        (lt==LoadType::PNG_INT) ? trainingPair.xTrain.push_back( MATH::flatten(getPng(pathInsert + line.substr(0,line.find(",")))) ) : trainingPair.xTrain.push_back( MATH::flatten(getPng(pathInsert + line.substr(line.find(",")+1,line.size()))) );
                        

                        if (mt == ModelType::CLASSIFY || mt == ModelType::BINARY) {
                            exVec = std::vector<double>(outputSize,0);
                            exVec[(int)score] = 1;

                            trainingPair.yTrain.push_back( exVec );
                        } else {
                            trainingPair.yTrain.push_back( std::vector<double>{score} );
                        }


                        if (cutOff && n > cutOff) {
                            break;
                        }                

                        n++;
                    }
                }

                break;
        
        }

    }


    std::tuple<std::vector<double>, std::vector<double>> pull(int n) const {
        std::string l;
        int c = 0;

        while (std::getline(*this->fp, l)){
            if (n == c) {
                break;
            }
            c++;
        }
                
        this->fp->clear();
        this->fp->seekg(0);

        return this->pullFunc(l, this->lt, this->mt, this->outputSize, this->pathInsert);
    }

    ~Loader() {
        this->fp->close();
        delete fp;
    }

};