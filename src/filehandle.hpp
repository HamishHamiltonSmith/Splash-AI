#pragma once

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <libpng16/png.h>
#include <vector>
#include <map>
#include <string>
#include <bitset>
#include <sstream>
#include <algorithm>

int width, height;
png_byte color_type;
png_byte bit_depth;
png_bytep* row_pointers = NULL;

//Permissible color types
// 1 - Greyscale: 1 val
// 2 - RGB: 3 vals
// 6 - RGBA - 4 vals
std::map<int, int> cTypes{{0,1}, {1,1}, {2,3}, {6,4}};


std::vector<std::vector<double>> getPng(std::string filename) {
    row_pointers=NULL;

    std::FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp) {abort();}

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) abort();

    png_infop info = png_create_info_struct(png);
    if(!info) abort();

    if(setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth = png_get_bit_depth(png, info);

    //A vector of ROWS
    //ie: imgGrey[row][coll]
    std::vector<std::vector<double>> imgGrey(height, std::vector<double>(width));

    if (row_pointers) abort();

    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }


    png_read_image(png, row_pointers);

    //Convert color type to values per pixel
    int valPerPixel=0;
    if (cTypes.find(color_type) != cTypes.end()){valPerPixel=cTypes.at(color_type);} else {abort();};

    
    //std::cout << "\n\n[";
    for (int y=0; y<height; y++) {
        //std::cout << "[";
        for (int x=0; x<width; x++) {
            //imgGrey[y][x] = (double)row_pointers[y][x] / 255;
            imgGrey[y][x] = (double)int(((double)row_pointers[y][x*valPerPixel]+(double)row_pointers[y][x*valPerPixel+1]+(double)row_pointers[y][x*valPerPixel+2])/3)/255;
            //std::cout << imgGrey[y][x] << ",";
        }
        //std::cout << "],\n";
    }
    //std::cout << "]";


    fclose(fp);
    return imgGrey;
}

std::vector<std::vector<double>> convolute(const std::vector<std::vector<double>>& image, std::vector<std::vector<double>> matrix) {
    std::vector<std::vector<double>> newImg(image.size()-2, std::vector<double>(image[0].size()-2));
    for (int y=1; y<image.size()-2; y++) {
        
        for (int x=1; x<image[0].size()-2; x++) {
            newImg[y][x] = int(matrix[0][0]*image[y-1][x-1]) + int(matrix[0][1]*image[y-1][x]) + int(matrix[0][2]*image[y-1][x+1]) +
            int(matrix[1][0]*image[y][x-1]) + int(matrix[1][1]*image[y][x]) + int(matrix[1][2]*image[y][x+1]) +
            int(matrix[2][0]*image[y+1][x-1]) + int(matrix[2][1]*image[y+1][x]) + int(matrix[2][2]*image[y+1][x+1]);
        }
    }

    return newImg;
} 

std::vector<std::vector<double>> maxPool(const std::vector<std::vector<double>>& image, int size) {
    int offX = 0;
    int offY = 0; 
    
    if (image.size()%size != 0) {
        offY++;
    } if (image[0].size()%size != 0){
        offX++;
    }

    std::vector<std::vector<double>> newImg((image.size()-offY)/size, std::vector<double>((image[0].size()-offX)/size));

    for (int y=offY; y<image.size()-1; y+=size) {
        for (int x=offX; x<image.at(0).size()-1; x+=size) {
            std::vector<double> vals{image.at(y).at(x), image.at(y).at(x+1), image.at(y+1).at(x), image.at(y+1).at(x+1)};
            newImg[y/(size+offY)][x/(size+offX)] = abs((double)(*std::max_element(vals.begin(), vals.end())));
        }
    }

    return newImg;
}

void saveToFile(std::string fname, std::string tokens) {
    std::ofstream file(fname);
    file << tokens;
    file.close();
}

std::string loadFromFile(std::string fname) {
    std::ifstream t(fname);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}
