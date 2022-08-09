#ifndef UTILS_H
#define UTILS_H

#include "MNISTDataLoader.h"
#include "NetworkModel.h"

struct MatShape
{
    unsigned int rows;
    unsigned int columns;

    MatShape(unsigned int _rows, unsigned int _columns)
    {
        rows = _rows;
        columns = _columns;
    }
};

struct FilterShape
{
    unsigned int in;
    unsigned int out;

    FilterShape(unsigned int _in, unsigned int _out)
    {
        in = _in;
        out = _out;
    }
};

namespace Evaluation
{
    void CalculateAccuracy(NetworkModel model, MNISTDataLoader testLoader);
}

#endif // !UTILS_H
