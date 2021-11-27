#include "./CrossEntropy.h"
#include<cmath>

double CrossEntropy::compute(const vector<double>& prediction, int expected_int) const
{
    double loss = 0.0;

    int output_layer_size = prediction.size();
    vector<double> expected_vector(output_layer_size, 0.0);
    expected_vector[expected_int] = 1.0;

    for (int it = 0; it < output_layer_size; ++it) {
        loss += -1 * expected_vector[it] * log(prediction[it]);
    }
    return loss;
}

vector<double> CrossEntropy::backward_start(const vector<double>& prediction, int expected_int) const
{
    int output_layer_size = prediction.size();
    vector<double> expected_vector(output_layer_size, 0.0);
    expected_vector[expected_int] = 1.0;

    vector<double> gradient_start(output_layer_size, 0.0);
    for (int i = 0; i < output_layer_size; ++i)
    {
        gradient_start[i] = -1 * expected_vector[i] / prediction[i];
        //cout << gradient_start[i] << " ";
    }
    return gradient_start;
}
