#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "Module.h"

class LeakyRelu : public Module
{
private:
    double negativeSlope = 1e-2;
    Tensor<double> LeakyReluFunc();
    Tensor<double> LeakyReluGradient();

public:
    LeakyRelu();
    virtual ~LeakyRelu() = default;
    void Compile(unique_ptr<Tensor<double>>& input) override;
    unique_ptr<Tensor<double>> ForwardPropagate(unique_ptr<Tensor<double>>& input) override;
    unique_ptr<Tensor<double>> BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate) override;
    void load(FILE* fileModel) override;
    void save(FILE* fileModel) override;
};

#endif // !LEAKYRELU_H
