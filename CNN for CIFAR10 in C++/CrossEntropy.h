#pragma once
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
class CrossEntropy
{
public:

    virtual double compute(const vector<double>& prediction,
        int expected_int) const;
    // backward error of output Layer
    virtual vector<double> backward_start(const vector<double>& prediction,
        int expected_int) const;
};
