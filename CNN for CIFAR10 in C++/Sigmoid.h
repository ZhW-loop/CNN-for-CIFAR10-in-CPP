#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
using namespace std;
// ÄÚÖÃµ½Fully Connected Layer
class Sigmoid {

public:

    // sigmoid(X) = [1.0/(1.0+exp(-X_i)) for i = 1..N]
    vector<double> compute(const vector<double>& X) const;

    // sigmoid'(X) = sigmoid(X)*(1-sigmoid(X))
    vector<double> gradient_a_to_Z(const vector<double>& X) const;

}; 