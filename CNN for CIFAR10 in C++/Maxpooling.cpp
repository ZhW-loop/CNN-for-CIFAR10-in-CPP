#include "./Maxpooling.h"

MaxPoolLayer::MaxPoolLayer(int k, int s)
{
    kernel_size = k; stride = s;
};
double max(double a, double b, double c, double d) {
    double out = a;
    if (b > out) out = b;
    if (c > out) out = c;
    if (d > out) out = d;
    return out;
}
Array3d MaxPoolLayer::compute(const Array3d& inputs) const
{
    int n_out = (inputs.height - kernel_size) / stride + 1;
    int m_out = (inputs.width - kernel_size) / stride + 1;

    vector<double> out;
    out.reserve(n_out * m_out * inputs.channel);
    for (int k = 0; k < inputs.channel; ++k) {
        for (int i = 0; i < n_out; ++i) {
            for (int j = 0; j < m_out; ++j) {
                out.push_back(max(inputs(kernel_size * i, kernel_size * j, k),
                    inputs(kernel_size * i, kernel_size * j + 1, k),
                    inputs(kernel_size * i + 1, kernel_size * j, k),
                    inputs(kernel_size * i + 1, kernel_size * j + 1, k)));
            }
        }
    }
    return Array3d(n_out, m_out, inputs.channel, out);
};

Array3d MaxPoolLayer::activate(const Array3d& Z) const
{
    return Z;
};

Array3d MaxPoolLayer::forward(const Array3d& inputs) const
{
    return compute(inputs);
};

Array3d MaxPoolLayer::gradient_L_to_Z(const Array3d& None, 
    const Array3d& gradient_L_to_a) const
{
    return gradient_L_to_a;
};
// 特殊，不使用本层参数(也没有)，使用input找到最大值的位置
// 也可以在forward pass过程中直接记录最大值的位置，其余位置直接填零
Array3d MaxPoolLayer::backward(const Array3d& gradient_L_to_Z, const Array3d& Z) const
{
    Array3d out(stride * (gradient_L_to_Z.height - 1) + kernel_size, stride * (gradient_L_to_Z.width - 1) + kernel_size, gradient_L_to_Z.channel);
    out.fill_with_zeros();
    for (int k = 0; k < gradient_L_to_Z.channel; ++k) {
        for (int i = 0; i < gradient_L_to_Z.height; ++i) {
            for (int j = 0; j < gradient_L_to_Z.width; ++j) {
                double _max = max(Z(kernel_size * i, kernel_size * j, k), Z(kernel_size * i, kernel_size * j + 1, k),
                    Z(kernel_size * i + 1, kernel_size * j, k), Z(kernel_size * i + 1, kernel_size * j + 1, k));
                if (Z(kernel_size * i, kernel_size * j, k) == _max) {
                    out(kernel_size * i, kernel_size * j, k) = gradient_L_to_Z(i, j, k);
                }
                else if (Z(kernel_size * i, kernel_size * j + 1, k) == _max) {
                    out(kernel_size * i, kernel_size * j + 1, k) = gradient_L_to_Z(i, j, k);
                }
                else if (Z(kernel_size * i + 1, kernel_size * j, k) == _max) {
                    out(kernel_size * i + 1, kernel_size * j, k) = gradient_L_to_Z(i, j, k);
                }
                else {
                    out(kernel_size * i + 1, kernel_size * j + 1, k) = gradient_L_to_Z(i, j, k);
                }
            }
        }
    }
    return out;
};
