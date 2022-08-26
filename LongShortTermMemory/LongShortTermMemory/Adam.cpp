#include "Adam.h"

Adam::Adam(double beta1, double beta2, double learningRate) : Optimizer(learningRate)
{
    m_Beta1 = beta1;
    m_Beta2 = beta2;
}

void Adam::InitializeV(map<string, Tensor<double>> parameters)
{
    map<string, Tensor<double>>::iterator it;
    for (it = parameters.begin(); it != parameters.end(); it++)
    {
        V[it->first] = Tensor<double>(it->second.numDims, it->second.m_Dims);
        V[it->first].zero();
    }    
}

void Adam::InitializeS(map<string, Tensor<double>> parameters)
{
    map<string, Tensor<double>>::iterator it;
    for (it = parameters.begin(); it != parameters.end(); it++)
    {
        S[it->first] = Tensor<double>(it->second.numDims, it->second.m_Dims);
        S[it->first].zero();
    }
}

void Adam::InitializeParams(map<string, Tensor<double>> parameters)
{
    InitializeV(parameters);
    InitializeS(parameters);
}

void Adam::UpdateParameters(map<string, Tensor<double>>& parameters, map<string, Tensor<double>> grads)
{
    map<string, Tensor<double>>::iterator itP;
    map<string, Tensor<double>>::iterator itG;
    for (itP = parameters.begin(), itG = grads.begin(); itP != parameters.end(), itG != grads.end(); itP++, itG++)
        V[itP->first] = (V[itP->first] * m_Beta1) + (itG->second * (1 - m_Beta1));

    for (itP = parameters.begin(), itG = grads.begin(); itP != parameters.end(), itG != grads.end(); itP++, itG++)
        S[itP->first] = (S[itP->first] * m_Beta2) + ((itG->second.pow(2)) * (1 - m_Beta2));

    for (itP = parameters.begin(); itP != parameters.end(); itP++)
        parameters[itP->first] = parameters[itP->first] - ((V[itP->first] / (S[itP->first].pow(0.5) + 1e-6)) * m_LearningRate);
}