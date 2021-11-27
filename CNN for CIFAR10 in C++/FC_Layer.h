#pragma once
#include "Array2d.h"
#include <vector>
#include "./Sigmoid.h"
using namespace std;
class FCLayer
{
private:
    Sigmoid Fun;
public:
    Array2d W;   // Unknow Parameters
    int num_in, num_out;
    FCLayer(int _num_in, int _num_out);
    void initialize();
    vector<double> forward(const vector<double>& inputs) const;
    vector<double> compute(const vector<double>& inputs) const;
    vector<double> activate(const vector<double>& z) const;
    vector<double> gradient_L_to_Z(const vector<double>& z,
        const vector<double>& backwrd_err) const;
    vector<double> backward(const vector<double>& gradient_L_to_Z) const;
    void update(const vector<double>& gradient_L_to_Z, const vector<double>& z,
        double lr);
    vector<double> get_learnable_parameters() const;
    void set_learnable_parameters(vector<double> learnable_parameters);
}; 