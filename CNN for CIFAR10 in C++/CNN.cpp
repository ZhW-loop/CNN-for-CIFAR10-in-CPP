#include "./CNN.h"
#include <assert.h>
#include <cassert>
#include <numeric>
using namespace std;
CNN::CNN(list<Layer3D*> _feature_detector, list<FCLayer> _classifier,
    CrossEntropy* _loss, double _feature_detector_lr, double _classifier_lr)
{
    feature_detector = _feature_detector;
    classifier = _classifier;
    loss = _loss;
    feature_detector_lr = _feature_detector_lr;
    classifier_lr = _classifier_lr;
    initialize();
}

void CNN::initialize()
{
    for (auto l : feature_detector) l->initialize();
    for (auto l : classifier) l.initialize();
}

void CNN::set_train_database(vector<Array3d>* images,
    vector<int>* labels)
{
    train_database_images = images;
    train_database_labels = labels;
};

void CNN::set_test_database(vector<Array3d>* images,
    vector<int>* labels)
{
    test_database_images = images;
    test_database_labels = labels;
}

vector<int> CNN::predict(vector<Array3d> inputs)
{
    vector<int> outputs;
    outputs.reserve(inputs.size());

    Array3d A_Layer3D;
    vector<double> a_Linear;

    // For all image in the inputs
    for (auto image : inputs) {

        A_Layer3D = image;
        for (auto l : feature_detector) A_Layer3D = l->forward(A_Layer3D);

        a_Linear = A_Layer3D.Flatten();
        for (auto l : classifier) a_Linear = l.forward(a_Linear);

        int output = std::distance(a_Linear.begin(),
            std::max_element(a_Linear.begin(), a_Linear.end()));

        outputs.push_back(output);

        for (int i = 0; i < a_Linear.size(); ++i) cout << a_Linear[i] << " ";
        cout << endl;
    }

    return outputs;
}

void CNN::train(int n_epoch)
{
    assert(!train_database_images->empty() && !train_database_labels->empty());

    // shuffle
    std::random_device rd;
    std::default_random_engine generator(rd());
    int s = train_database_images->size();
    vector<int > rand_indxs(s);
    for (int i = 0; i < s; ++i) rand_indxs[i] = i;

    std::cout << "Start training for " << n_epoch << " epochs" << std::endl;

    // zs,as存储正向传播时的Z和a
    stack<vector<double> > zs_Linear, as_Linear;
    // a,Z,backward error,layer error
    vector<double> a_Linear, Z_Linear, gradient_L_to_a_Linear, gradient_L_to_Z_Linear;
    stack<Array3d> zs_Layer3D, as_Layer3D;
    Array3d a_Layer3D, Z_Layer3D, gradient_L_to_a_Layer3D, gradient_L_to_Z_Layer3D;

    for (int n = 0; n < n_epoch; ++n) {
        std::cout << "\r" << "Epoch " << n + 1 << endl;

        std::shuffle(rand_indxs.begin(), rand_indxs.end(),
            generator);

        int cnt = 0;
        for (const auto& i : rand_indxs) {
            a_Layer3D = (*train_database_images)[i];

            //forward pass ,save a and Z
            as_Layer3D.push(a_Layer3D);
            for (auto l : feature_detector) {
                Z_Layer3D = l->compute(a_Layer3D);
                zs_Layer3D.push(Z_Layer3D);
                a_Layer3D = l->activate(Z_Layer3D);
                as_Layer3D.push(a_Layer3D);
            }
            as_Layer3D.pop();
            a_Linear = a_Layer3D.Flatten();

            as_Linear.push(a_Linear);
            for (auto& l : classifier) {
                Z_Linear = l.compute(a_Linear);
                zs_Linear.push(Z_Linear);
                a_Linear = l.activate(Z_Linear);
                as_Linear.push(a_Linear);
            }
            as_Linear.pop();

            // Compute the backward error of the output layer
            gradient_L_to_a_Linear = loss->backward_start(a_Linear, (*train_database_labels)[i]);

            // 使用本层的Z获得本层的Layer Error，使用上一层的a进行update
            // 使用本层的Unknown Parameter获得上一层的Backward Error(Maxpooling特殊)
            for (list<FCLayer>::reverse_iterator l = classifier.rbegin();
                l != classifier.rend(); ++l) {
                gradient_L_to_Z_Linear = l->gradient_L_to_Z(zs_Linear.top(), gradient_L_to_a_Linear);
                zs_Linear.pop();
                gradient_L_to_a_Linear = l->backward(gradient_L_to_Z_Linear);
                l->update(gradient_L_to_Z_Linear, as_Linear.top(), classifier_lr);
                as_Linear.pop();
            }
            //zs_Linear.pop();
            assert(zs_Linear.empty());
            assert(as_Linear.empty());

            Array3d tmp(zs_Layer3D.top().height, zs_Layer3D.top().width,
                zs_Layer3D.top().channel, gradient_L_to_a_Linear);
            gradient_L_to_a_Layer3D = tmp;
            
            for (list<Layer3D*>::reverse_iterator l = feature_detector.rbegin();
                l != feature_detector.rend(); ++l) {
                gradient_L_to_Z_Layer3D = (*l)->gradient_L_to_Z(zs_Layer3D.top(), gradient_L_to_a_Layer3D);
                zs_Layer3D.pop();
                gradient_L_to_a_Layer3D = (*l)->backward(gradient_L_to_Z_Layer3D, as_Layer3D.top());
                (*l)->update(gradient_L_to_Z_Layer3D, as_Layer3D.top(), feature_detector_lr);
                as_Layer3D.pop();
            }
            assert(as_Layer3D.empty());
            assert(zs_Layer3D.empty());
            cnt++;
            cout << "Image " << cnt << " finished" << endl;
        }
        cout << "Accuracy on test dataset: " << test_accuracy() << endl;
        //cout << (*this)<<endl;
    }
}

double CNN::test_accuracy()
{
    assert(!test_database_images->empty() && !test_database_labels->empty());

    // We compute the predicted label for all the images in the test dataset
    vector<int> out = predict(*test_database_images);

    
    int size = test_database_labels->size();
    int count_good_pred = 0;
    for (int it = 0; it < size; ++it) {
        if (out[it] == (*test_database_labels)[it]) {
            ++count_good_pred;
        }
        cout << "Predict: " << out[it] << endl;
        cout << "Label: " << (*test_database_labels)[it] << endl;
    }

    return (double)count_good_pred / (double)size;
}

ostream& operator<<(ostream& os, const CNN& net)
{
   
    for (auto l : net.feature_detector) {
        if (l->is_learnable()) {
            for (auto vec : l->get_learnable_parameters()) {
                for (auto param : vec) {
                    os << param << '\n';
                }
            }
        }
    }
    
    for (auto l : net.classifier) {
        for (auto parameter : l.get_learnable_parameters()) {
            os << parameter << '\n';
        }
    }

    return os;
}

void CNN::save(string name)
{
    std::ofstream ofs(name + ".txt");
    ofs << (*this);
    ofs.close();
}