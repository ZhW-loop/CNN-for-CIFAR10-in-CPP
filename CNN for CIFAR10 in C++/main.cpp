#include <iostream>
#include <list>
#include <ctime>
#include "./CNN.h"
#include "./CrossEntropy.h"
using namespace std;

int main()
{

    // path to database
    string filename_train_images = "./CIFAR10_for_C++_train_image";
    string filename_train_labels = "./CIFAR10_for_C++_train_label";
    string filename_test_images = "./CIFAR10_for_C++_test_image";
    string filename_test_labels = "./CIFAR10_for_C++_test_label";

    // ÍøÂç½á¹¹
    Layer3D* f1 = new ConvLayer(5, 3, 32, 1, 2);
    Layer3D* p1 = new MaxPoolLayer(2, 2);
    Layer3D* f2 = new ConvLayer(5, 32, 32, 1, 2);
    Layer3D* p2 = new MaxPoolLayer(2, 2);
    Layer3D* f3 = new ConvLayer(5, 32, 64, 1, 2);
    Layer3D* p3 = new MaxPoolLayer(2, 2);
    list<Layer3D*> feature_detector{ f1, p1, f2, p2, f3, p3 };

    FCLayer l1(4 * 4 * 64, 64);
    FCLayer l2(64, 10);
    list<FCLayer> classifier{ l1, l2 };

    // Learning rates
    double classifier_lr = 0.03;
    double feature_detector_lr = 0.3;

    //CrossEntropy* loss = new CrossEntropy;
    CrossEntropy* loss = new CrossEntropy;
    // Number of epochs
    int n_epoch = 10;


    // Building the network
    CNN net(feature_detector, classifier, loss,
        feature_detector_lr, classifier_lr);

    // Read and set the database
    Data_Handler database_loader;
    vector<Array3d> train_database_images =
        database_loader.read_CIFAR10_image(filename_train_images);
    vector<int> train_database_labels =
        database_loader.read_CIFAR10_label(filename_train_labels);
    vector<Array3d> test_database_images =
        database_loader.read_CIFAR10_image(filename_test_images);
    vector<int> test_database_labels =
        database_loader.read_CIFAR10_label(filename_test_labels);
    net.set_train_database(&test_database_images, &test_database_labels);
    net.set_test_database(&test_database_images, &test_database_labels);

    //net.test_accuracy();
    net.train(n_epoch);

    // Accuracy test
    //cout << "Accuracy on test dataset: " << net.test_accuracy() << endl;
    //net.save("notrain");

    for (auto pointer : feature_detector) delete pointer;
    return 0;
}