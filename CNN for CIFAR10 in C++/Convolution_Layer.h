#pragma once
#include <iostream>
#include "./Layer3D.h"
#include "./ReLU.h"
#include <vector>
class ConvLayer : public Layer3D
{
private:
    ReLU Fun;
    int kernel_size, out_channel, stride, size_padding;
    int in_channel;
public:
    vector<Array3d> kernels;
    ConvLayer(int, int, int, int, int);
    virtual ~ConvLayer() {};
    virtual void initialize();
    // foward pass
    virtual Array3d compute(const Array3d&) const;
    virtual Array3d activate(const Array3d&) const;
    virtual Array3d forward(const Array3d&) const;
    // Layer Error
    virtual Array3d gradient_L_to_Z(const Array3d&, const Array3d&) const;
    // backward error of pre Layer
    virtual Array3d backward(const Array3d&, const Array3d&) const;
    // dW
    virtual void update(const Array3d&, const Array3d&, double);
    virtual vector<vector<double> > get_learnable_parameters() const;
    virtual void set_learnable_parameters(vector<vector<double> >);
    virtual bool is_learnable() const { return true; };
};
