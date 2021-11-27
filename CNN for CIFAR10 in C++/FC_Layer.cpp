#include "FC_Layer.h"

FCLayer::FCLayer(int _num_in, int _num_out)
{
    num_in = _num_in;
    num_out = _num_out;
    initialize();
};

void FCLayer::initialize()
{
    W = Array2d(num_out, num_in);
    double out = double(num_out);
    double in = double(num_in);
    W.fill_with_random_normal(0.0, 2.0 / (in + out));
};

vector<double> FCLayer::activate(const vector<double>& z) const
{
    return Fun.compute(z);
};

vector<double> FCLayer::compute(const vector<double>& inputs) const
{
    return W.dot(inputs);
};

vector<double> FCLayer::forward(const vector<double>& inputs) const
{
    return activate(compute(inputs));
};

vector<double> FCLayer::gradient_L_to_Z(const vector<double>& z,
    const vector<double>& gradient_L_to_a) const
{
    vector<double> out = Fun.gradient_a_to_Z(z);
    std::transform(out.begin(), out.end(), gradient_L_to_a.begin(), out.begin(),
        multiplies<double>());
    return out;
};

// 求对上一层输出的偏导Backward Error，未知参数矩阵转置后按行点乘Layer Error即可
vector<double> FCLayer::backward(const vector<double>& gradient_L_to_Z) const
{
    return W.Tdot(gradient_L_to_Z);
};

// 求dW，Layer Error中每个数乘所有输入a'更新对应的w
void FCLayer::update(const vector<double>& gradient_L_to_Z, 
    const vector<double>& inputs, double lr)
{
    int gradient_size = gradient_L_to_Z.size();
    int z_size = inputs.size();
    for (int i = 0; i < gradient_size; ++i)
        for (int j = 0; j < z_size; ++j) 
            W.values[i * num_in + j] -= gradient_L_to_Z[i] * inputs[j] * lr;
};

vector<double> FCLayer::get_learnable_parameters() const
{
    return W.values;
}

void FCLayer::set_learnable_parameters(vector<double> learnable_parameters)
{
    W.values = learnable_parameters;
}
