#pragma once
#include <vector>
#include <fstream>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <./Array3d.h>
using namespace std;

class Data_Handler
{

    public:
        // processed by Pytorch
        vector<Array3d> read_CIFAR10_image(string filename) const;
        vector<int> read_CIFAR10_label(string filename) const;

}; 

void show_image(Array3d);

