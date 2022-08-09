#include "Utils.h"

namespace Evaluation
{
    void CalculateAccuracy(NetworkModel model, MNISTDataLoader testLoader)
    {
        int hits = 0;
        int total = 0;
        printf("Testing...\n");
        int num_test_batches = testLoader.GetNumBatches();
        for (int i = 0; i < num_test_batches; ++i) {
            if ((i + 1) % 10 == 0 || i == (num_test_batches - 1))
            {
                printf("\rIteration %d/%d", i + 1, num_test_batches);
                fflush(stdout);
            }
            pair<Tensor<double>, vector<int> > xy = testLoader.NextBatch();
            vector<int> predictions = model.predict(xy.first);
            for (int j = 0; j < predictions.size(); ++j) {
                if (predictions[j] == xy.second[j]) {
                    hits++;
                }
            }
            total += xy.second.size();
        }
        printf("\n");

        printf("Accuracy: %.2f%% (%d/%d)\n", ((double)hits * 100) / total, hits, total);
    }
}