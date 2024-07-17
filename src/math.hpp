#pragma once
#include <vector>
#include <algorithm>
#include <iostream>

namespace MATH {
    //Utilities
    std::vector<double> flatten(std::vector<std::vector<double>> v) {
        std::vector<double> result;
        std::for_each(v.begin(), v.end(), [&](std::vector<double> inner){result.insert(result.end(), inner.begin(), inner.end());});
        return result;
    }

    //Naitive vector class
    template<typename T> class Vector {
        public:
        std::vector<T> core;
        
        Vector(int n, T value) {
            this->core = std::vector<T>(n, value);
        }

        Vector() {
            this->core = std::vector<T>();
        }

        Vector(std::vector<T> vec) {
            this->core = vec;
        }

        void push_back(T val) {
            core.push_back(val);
        }

        void clear() {
            core.clear();
        }

        void zero() {
            core = std::vector<T>(core.size(), 0);
            std::fill(core.begin(), core.end(), 0);
        }

        T& at(int idx) {
            return this->core.at(idx);
        }

        double sum() {
            double total = 0;
            for (auto element: this->core) {
                total += element;
            }
            return total;
        }

        Vector<T> subtract(Vector<T> vec) {
            Vector<T> out;
            for (int x=0; x<this->core.size(); x++) {
                out.push_back(this->core.at(x) - vec.at(x));
            }

            return out;
        }

        Vector<T> operator* (const double& scalar) {
            Vector<T> out;
            for (auto& element: this->core) {
                out.push_back(element*scalar);
            }

            return out;
        }

        Vector<T> operator* (Vector<T>& vec) {
            Vector<T> out;
            for (int x=0; x<this->core.size(); x++) {
                out.push_back(this->core.at(x) * vec.at(x));
            }

            return out;
        }

        Vector<T> operator+ (const double& scalar) {
            Vector<T> out;
            for (int x=0; x<this->core.size(); x++) {
            out.push_back(this->core.at(x) + scalar);
            }

            return out;
        }

        


        void print() {
            int c = 0;
            std::cout << "{";
            for (const T& val: this->core) {
                std::cout << val << std::endl;
                if (c==10) {
                    //std::cout << "\n";
                    c=0;
                } else {
                    c++;
                }
                std::cout << ",";
            }
            std::cout << "}";
        }
    };
}


