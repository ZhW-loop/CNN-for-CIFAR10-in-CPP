#include"./Sigmoid.h"

vector<double> Sigmoid::compute(const vector<double>& X) const
{
    vector<double> out;
    for (auto x : X) out.push_back(1.0 / (1.0 + exp(x)));
    return out;
}
// a对Z求偏导，需要forward pass中的Z
vector<double> Sigmoid::gradient_a_to_Z(const vector<double>& X) const
{
    vector<double> out = compute(X);
    for (int i = 0; i < out.size(); ++i)
        out[i] = out[i] * (1 - out[i]);
    return out;
}