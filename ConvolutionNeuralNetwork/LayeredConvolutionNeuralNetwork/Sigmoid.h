#ifndef SIGMOID_H
#define SIGMOID_H

#include "Module.h"

/*
 * Sigmoid activation layer
 * Output: 1.0 / (1.0 + exp(-x))
 */

class Sigmoid : public Module
{
private:
    Tensor<double> SigmoidFunc();
    Tensor<double> SigmoidGradient();

public:
    Sigmoid();
    virtual ~Sigmoid() = default;
    void Compile(unique_ptr<Tensor<double>>& input) override;
    unique_ptr<Tensor<double>> ForwardPropagate(unique_ptr<Tensor<double>>& input) override;
    unique_ptr<Tensor<double>> BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate) override;
    void load(FILE* fileModel) override;
    void save(FILE* fileModel) override;
};

#endif // !SIGMOID_H
