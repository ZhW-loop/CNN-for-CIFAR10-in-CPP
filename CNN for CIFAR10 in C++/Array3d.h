#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <assert.h>

using namespace std;

class Array3d {

public:

    int height,  width,  channel;
    vector<double> values;

    Array3d(int _height, int _width, int _channel);

    Array3d() { height = 0; width = 0; channel = 0; };
    Array3d(const Array3d& A)
    {
        height = A.height; width = A.width; channel = A.channel; values = A.values;
    };
    Array3d(int _height, int _width, int _channel, vector<double> _values)
    {
        height = _height; width = _width; channel = _channel; values = _values;
    };
    Array3d operator = (const Array3d& A)
    {
        height = A.height; width = A.width; channel = A.channel; values = A.values;
        return *this;
    };
    void fill_with_random_normal(double mean, double var);

    void fill_with_zeros();

    inline double operator()(int i, int j, int k) const
    {
        return values[height * width * k + width * i + j];
    };

    inline double& operator()(int i, int j, int k)
    {
        return values[height * width * k + width * i + j];
    };

    vector<double> Flatten() const { return values; };

    // 以(i, j)为左上顶点作卷积操作
    double convolution(const Array3d& kernel, int start_i, int start_j) const;

    // forward pass 和 dW-update使用
    void zero_padding(int size_padding);

    // backward error使用
    void clear_padding(int size_padding);

    Array3d& operator *= (double x);

    Array3d& operator -= (const Array3d& other_array);

    Array3d& operator += (const Array3d& other_array);

    Array3d operator * (const Array3d& other_array) const;

    // 旋转180° 未使用
    Array3d rotate180();

    void print();

}; 

