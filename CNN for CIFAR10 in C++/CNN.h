#include <iostream>
#include <list>
#include <stack>
#include <iterator>
#include <ctime>
#include <fstream>
#pragma once
#include <assert.h>
#include "./Array3d.h"
#include "./db_handler.h"
#include "./CrossEntropy.h"
#include "./FC_Layer.h"
#include "./Convolution_Layer.h"
#include "./Maxpooling.h"
#include "./MSE.h"
using namespace std;
class CNN
{

    private:
        list<Layer3D*> feature_detector;
        list<FCLayer> classifier;
        CrossEntropy* loss;
        double classifier_lr, feature_detector_lr; //learning rates
        vector<Array3d>* train_database_images;
        vector<int >* train_database_labels;
        vector<Array3d>* test_database_images;
        vector<int >* test_database_labels;
    public:
        CNN(list<Layer3D*> _feature_detector, list<FCLayer> _classifier,
            CrossEntropy* _loss, double _feature_detector_lr, double _classifier_lr);

        void initialize();

        void set_train_database(vector<Array3d>* images, vector<int >* labels);
        void set_test_database(vector<Array3d>* images, vector<int >* labels);

        vector<int> predict(vector<Array3d> inputs);

        // Train the network for n_peoch epochs on the train database
        void train(int n_peoch);

        // Test accuracy of the network on the test database with predict
        double test_accuracy();

        friend ostream& operator<<(ostream& os, const CNN& net);
        void save(string name);

};


