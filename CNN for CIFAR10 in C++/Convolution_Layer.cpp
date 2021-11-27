#include "Convolution_Layer.h"
using namespace std;
ConvLayer::ConvLayer(int k, int i, int o, int s, int p)
{
    kernel_size = k;
    in_channel = i;
    out_channel = o;
    stride = s;
    size_padding = p;
}

void ConvLayer::initialize()
{
    for (int it = 0; it < out_channel; ++it) {
        Array3d Filter(kernel_size, kernel_size, in_channel);
        Filter.fill_with_random_normal(0.0, 3.0 / (2.0 * kernel_size + in_channel));
        kernels.push_back(Filter);
    }
}

Array3d ConvLayer::compute(const Array3d& inputs) const
{   
    int n_out = (inputs.height - kernel_size + 2 * size_padding) / stride + 1;
    int m_out = (inputs.width - kernel_size + 2 * size_padding) / stride + 1;
    Array3d out(n_out, m_out, out_channel);
    out.fill_with_zeros();
    Array3d inputs_padding = inputs;
    inputs_padding.zero_padding(size_padding);

    int start_i, start_j;
    for (int i = 0; i < n_out; ++i) {
        start_i = i * stride;
        for (int j = 0; j < m_out; ++j) {
            start_j = j * stride;
            for (int k = 0; k < out_channel; ++k) {
                out(i, j, k) = inputs_padding.convolution(kernels[k], start_i, start_j);
            }
        }
    }
    return out;
}

Array3d ConvLayer::activate(const Array3d& Z) const
{
    return Fun.compute(Z);
}

Array3d ConvLayer::forward(const Array3d& inputs) const
{
    return activate(compute(inputs));
}

Array3d ConvLayer::gradient_L_to_Z(const Array3d& Z,
    const Array3d& gradient_L_to_a) const
{
    return gradient_L_to_a * Fun.gradient_a_to_Z(Z);
}

// 未使用卷积操作
Array3d ConvLayer::backward(const Array3d& gradient_L_to_Z, const Array3d& None) const
{
    int height = gradient_L_to_Z.height;
    int width = gradient_L_to_Z.width;

    Array3d out(stride * (gradient_L_to_Z.height - 1) + kernel_size,
        stride * (gradient_L_to_Z.width - 1) + kernel_size,
        in_channel);
    out.fill_with_zeros();
    int h_shift, w_shift, x_out, y_out;
    double g;
    Array3d filter;

    for (int r = 0; r < out_channel; ++r) {
        filter = kernels[r];
        for (int i = 0; i < height; ++i) {
            h_shift = i * stride;
            for (int j = 0; j < width; ++j) {
                w_shift = j * stride;
                g = gradient_L_to_Z(i, j, r);
                for (int p = 0; p < kernel_size; ++p) {
                    x_out = p + h_shift;
                    for (int q = 0; q < kernel_size; ++q) {
                        y_out = q + w_shift;
                        for (int t = 0; t < in_channel; ++t) {
                            out(x_out, y_out, t) += filter(p, q, t) * g;
                        }
                    }
                }
            }
        }
    }
    out.clear_padding(size_padding);
    return out;
}
// 未使用卷积操作
void ConvLayer::update(const Array3d& gradient_L_to_Z, const Array3d& inputs, double lr)
{
    Array3d inputs_padding = inputs;
    inputs_padding.zero_padding(size_padding);
    double temp;
    int x_out;
    int height = gradient_L_to_Z.height;
    int width = gradient_L_to_Z.width;
    for (int p = 0; p < kernel_size; ++p) {
        for (int q = 0; q < kernel_size; ++q) {
            for (int t = 0; t < in_channel; ++t) {
                for (int r = 0; r < out_channel; ++r) {
                    temp = 0;
                    for (int i = 0; i < height; ++i) {
                        x_out = p + i * stride;
                        for (int j = 0; j < width; ++j) {
                            temp += inputs_padding(x_out, q + j * stride, t) * gradient_L_to_Z(i, j, r);
                        }
                    }
                    kernels[r](p, q, t) -= lr * temp;
                }
            }
        }
    }
    int i = 0;
    i++;
}


vector<vector<double> > ConvLayer::get_learnable_parameters() const
{
    vector<vector<double> > parameters;
    for (auto Filter : kernels) {
        parameters.push_back(Filter.values);
    }
    return parameters;
}

void ConvLayer::set_learnable_parameters(vector<vector<double> > parameters)
{
    for (int i = 0; i < out_channel; ++i) {
        kernels[i].values = parameters[i];
    }
}