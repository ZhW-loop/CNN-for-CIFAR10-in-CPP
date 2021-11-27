#include "./MSE.h"

double MSE::compute(const vector<double>& prediction,
    int8_t expected_int) const
{
    double loss = 0.0;
    int output_layer_size = prediction.size();
    vector<double> expected_vector(output_layer_size, 0.0);
    expected_vector[expected_int] = 1.0;

    for (int it = 0; it < output_layer_size; ++it) {
        loss += std::pow(prediction[it] - expected_vector[it], 2);
    }

    return loss;
}

vector<double> MSE::backward_start(const vector<double>& prediction,
    int8_t expected_int) const
{
    int output_layer_size = prediction.size();
    vector<double> expected_vector(output_layer_size, 0.0);
    expected_vector[expected_int] = 1.0;

    transform(prediction.begin(), prediction.end(),
        expected_vector.begin(), expected_vector.begin(),
        std::minus<double>());

    return expected_vector;
}
