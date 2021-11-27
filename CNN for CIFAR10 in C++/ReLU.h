#pragma once
#include <iostream>
#include "Array3d.h"
#include <algorithm>
using namespace std;
// ÄÚÖÃµ½Convolution Layer
class ReLU {
public:

    // ReLU(X) = max(0,X)  
    Array3d compute(const Array3d& X) const;

    // ReLU'(X) = [1 iF Xi > 0 else 0 for i = 1..N]
    Array3d gradient_a_to_Z(const Array3d& X) const;
}; 