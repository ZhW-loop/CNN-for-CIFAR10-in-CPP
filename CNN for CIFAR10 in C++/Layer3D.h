#pragma once
#include <iostream>
#include "Array3d.h"
#include <vector>
using namespace std;

class Layer3D
{
public:
    virtual void initialize() = 0;
    virtual ~Layer3D() {};
    virtual Array3d compute(const Array3d&) const = 0;
    virtual Array3d activate(const Array3d&) const = 0;
    virtual Array3d forward(const Array3d&) const = 0;
    virtual Array3d gradient_L_to_Z(const Array3d&, const Array3d&) const = 0;
    virtual Array3d backward(const Array3d&, const Array3d&) const = 0;
    virtual void update(const Array3d&, const Array3d&, double) = 0;
    virtual vector<vector<double> > get_learnable_parameters() const = 0;
    virtual void set_learnable_parameters(vector<vector<double> >) = 0;
    virtual bool is_learnable() const = 0;
};