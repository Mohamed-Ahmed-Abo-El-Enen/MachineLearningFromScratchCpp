#include "GradientDescent.h"

GradientDescent::GradientDescent(double learningRate) : Optimizer(learningRate)
{
}

void GradientDescent::UpdateParameters(map<string, Tensor<double>>& parameters, map<string, Tensor<double>> grads)
{
    map<string, Tensor<double>>::iterator itP;
    map<string, Tensor<double>>::iterator itG;
    for (itP = parameters.begin(), itG = grads.begin(); itP != parameters.end(), itG != grads.end(); itP++, itG++)
        parameters[itP->first] = parameters[itP->first] - (itG->second * m_LearningRate);        
}
void GradientDescent::InitializeParams(map<string, Tensor<double>> parameters)
{

}