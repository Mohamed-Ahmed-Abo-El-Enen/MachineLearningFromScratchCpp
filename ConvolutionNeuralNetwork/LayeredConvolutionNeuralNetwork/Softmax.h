#include "OutputLayer.h"
/*
 * Applies softmax and uses cross entropy as loss function
 */
class Softmax : public OutputLayer
{
private:
    unique_ptr<Tensor<double>> GetSoftmaxProbabilities(unique_ptr<Tensor<double>>& input);
    double CrossEntropy(unique_ptr<Tensor<double>>& yHat, vector<int>& yTrue);
    unique_ptr<Tensor<double>> CrossEntropyPrime(unique_ptr<Tensor<double>>& yHat, vector<int> yTrue);
    double epslion = 0.0000000001;

public:
    Softmax();
    virtual ~Softmax() = default;
    unique_ptr<Tensor<double>> predict(unique_ptr<Tensor<double>>& input) override;
    pair<double, unique_ptr<Tensor<double>>> BackwardPropagate(unique_ptr<vector<int>>& yTrue) override;
    void Compile(unique_ptr<Tensor<double>>& input) override;
};