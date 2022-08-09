#ifndef TANH_H
#define TANH_H

#include <math.h>
#include "Module.h"

class Tanh : public Module
{
private:
    Tensor<double> TanhFunc();
    Tensor<double> TanhGradient();

public:
    Tanh();
    virtual ~Tanh() = default;
    void Compile(unique_ptr<Tensor<double>>& input) override;
    unique_ptr<Tensor<double>> ForwardPropagate(unique_ptr<Tensor<double>>& input) override;
    unique_ptr<Tensor<double>> BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate) override;
    void load(FILE* fileModel) override;
    void save(FILE* fileModel) override;
};

#endif // !TANH_H
