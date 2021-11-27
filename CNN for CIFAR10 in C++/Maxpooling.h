#pragma once
#include "./Layer3D.h"
class MaxPoolLayer : public Layer3D
{
private:
    int kernel_size, stride;
public:
    MaxPoolLayer(int, int);
    virtual ~MaxPoolLayer() {};
    virtual void initialize() {};
    virtual Array3d compute(const Array3d&) const;
    virtual Array3d activate(const Array3d&) const;
    virtual Array3d forward(const Array3d&) const;
    // a = Z
    virtual Array3d gradient_L_to_Z(const Array3d&, const Array3d&) const;
    // 特殊，需要使用input
    virtual Array3d backward(const Array3d&, const Array3d&) const;
    // 仅继承
    virtual void update(const Array3d&, const Array3d&, double) {};
    virtual vector<vector<double> > get_learnable_parameters() const
    {
        vector<vector<double> > v; return v;
    };
    virtual void set_learnable_parameters(vector<vector<double> > v) {};
    virtual bool is_learnable() const { return false; };
};
