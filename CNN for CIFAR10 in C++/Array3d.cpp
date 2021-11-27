#include "Array3d.h"
Array3d::Array3d(int rows, int cols, int h)
{
    height = rows;
    width = cols;
    channel = h;
}

void Array3d::fill_with_zeros() {
    int size = height * width * channel;
    values.reserve(size);
    for (int it = 0; it < size; it++) {
        values.push_back(0);
    };
}

void Array3d::fill_with_random_normal(double mean, double var)
{
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(mean, var);
    values.clear();
    int size = height * width * channel;
    values.reserve(size);
    for (int i = 0; i < size; i++) {
        values.push_back(distribution(generator));
    }
}

void Array3d::zero_padding(int size_padding)
{   
    if (size_padding == 0) return;
    vector<double> tmp_values = values;
    values.clear();
    int tmp_cnt = 0;
    for (int k = 0; k < channel; ++k)
    {
        for (int i = 0; i < size_padding * (width + size_padding * 2); ++i)
            values.push_back(0.0);
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < size_padding; ++j) values.push_back(0.0);
            for (int j = 0; j < width; ++j) values.push_back(tmp_values[tmp_cnt++]);
            for (int j = 0; j < size_padding; ++j) values.push_back(0.0);
        }
        for (int i = 0; i < size_padding * (width + size_padding * 2); ++i)
            values.push_back(0.0);
    }
    height = height + size_padding * 2;
    width = width + size_padding * 2;
    channel = channel;
}

void Array3d::clear_padding(int size_padding)
{
    if (size_padding == 0) return;
    height = height - size_padding * 2;
    width = width - size_padding * 2;
    channel = channel;
    vector<double> tmp_values = values;
    values.clear();
    int tmp_cnt = 0;
    for (int k = 0; k < channel; ++k)
    {
        for (int i = 0; i < size_padding * (width + size_padding * 2); ++i)
            tmp_cnt++;
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < size_padding; ++j) tmp_cnt++;
            for (int j = 0; j < width; ++j) values.push_back(tmp_values[tmp_cnt++]);
            for (int j = 0; j < size_padding; ++j) tmp_cnt++;
        }
        for (int i = 0; i < size_padding * (width + size_padding * 2); ++i)
            tmp_cnt++;
    }
}

Array3d& Array3d::operator*=(double d)
{
    for (int i = 0; i < values.size(); ++i)
        values[i] *= d;
    return *this;
};

Array3d& Array3d::operator-=(const Array3d& A)
{
    assert((A.height == height) && (A.width == width) && (A.channel == channel));

    transform(values.begin(), values.end(), A.values.begin(), values.begin(),
        minus<double>());
    return *this;
};

Array3d& Array3d::operator+=(const Array3d& A)
{
    assert((A.height == height) && (A.width == width) && (A.channel == channel));

    transform(values.begin(), values.end(), A.values.begin(), values.begin(),
        plus<double>());
    return *this;
};

Array3d Array3d::operator*(const Array3d& A) const
{
    assert((A.height == height) && (A.width == width) && (A.channel == channel));

    Array3d out(*this);
    transform(out.values.begin(), out.values.end(), A.values.begin(),
        out.values.begin(), multiplies<double>());
    return out;
}
Array3d Array3d::rotate180()
{
    Array3d temp;
    temp = *this;
    for (int k = 0; k < channel; ++k)
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
                temp(i, j, k) = (*this)(height - i - 1, width - j - 1, k);
    return temp;
}
double Array3d::convolution(const Array3d& kernel, int start_i, int start_j) const
{
    assert((kernel.channel == this->channel));
    assert((start_i + kernel.height <= height) && (start_j + kernel.width <= width));

    double res = 0.0;
    for (int k = 0; k < channel; ++k) {
        for (int i = 0; i < kernel.height; ++i) {
            for (int j = 0; j < kernel.width; ++j) {
                res += kernel(i, j, k) * (*this)(i + start_i, j + start_j, k);
            }
        }
    }
    return res;
}

void Array3d::print()
{
    for (int k = 0; k < channel; ++k) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                std::cout << values[k * width * height + i * width + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "-----" << endl;
    }
};