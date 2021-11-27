#include "./ReLU.h"


Array3d ReLU::compute(const Array3d& X) const
{
    vector<double> out_values;
    int out_size = X.values.size();
    out_values.reserve(out_size);
    double val;
    for (int it = 0; it < out_size; ++it) {
        val = X.values[it];
        out_values.push_back((val >= 0.0) ? val : 0.0);
    }
    return Array3d(X.height, X.width, X.channel, out_values);
}
// a对Z求偏导，需要forward pass中的Z
Array3d ReLU::gradient_a_to_Z(const Array3d& X) const
{
    vector<double> out_values;
    int out_size = X.values.size();
    out_values.reserve(out_size);
    for (int it = 0; it < out_size; ++it) {
        out_values.push_back((X.values[it] >= 0.0) ? 1.0 : 0.0);
    }
    return Array3d(X.height, X.width, X.channel, out_values);
}