#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <assert.h>
class Array2d {

public:
    int num_rows, num_cols;
    std::vector<double> values;

    Array2d() { num_rows = 0; num_cols = 0; };
    Array2d(int _num_rows, int _num_cols);
    void print() const;

    Array2d& operator*=(double x);
    Array2d& operator-=(const Array2d& other_array);

    void fill_with_zeros();
    void fill_with_random_normal(double mean, double var);

    // 行点积
    std::vector<double> dot(const std::vector<double>& extern_vector) const;

    // transpose and dot 列点积
    std::vector<double> Tdot(const std::vector<double>& extern_vector) const;

}; 
