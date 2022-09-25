#include "HiddenMarkovModel.h"
#include "TestData.h"
#include "HHMAlgorithm.h"
#include "Estimator.h"

void Test1()
{
	ifstream modelSource("../Model/CustomDefault.model");
	ifstream dataSource("../Dataset/CustomDefault.data");

	if (!modelSource.good())
	{
		cerr << "ERROR: failed to open model file properly. " << endl;
		return;
	}

	if (!dataSource.good())
	{
		cerr << "ERROR: failed to open data file properly. " << endl;
		return;
	}

	// enable exceptions to signal errors later while reading model and data
	ios_base::iostate ioExcept = (ifstream::failbit | ifstream::badbit | ifstream::eofbit);

	modelSource.exceptions(ioExcept);
	dataSource.exceptions(ioExcept);

	// section: read model and data
	HiddenMarkovModel model;
	TestData data;

	try
	{
		model.ReadModel(modelSource);
	}
	catch (const exception & e)
	{
		cerr << "ERROR: Fatal problem while reading model. Details: '" << e.what() << "'" << endl;
		return;
	}

	try
	{
		data.ReadTestData(model, dataSource);
	}
	catch (const exception & e)
	{
		cerr << "ERROR: Fatal problem while reading experiment data. Details: '" << e.what() << "'" << endl;
		return;
	}

	// secton: run and estimate viterbi predictions
	HHMAlgorithm hhmAlgo;
	Estimator estimator;
	vector<size_t> mostProbableSeq = hhmAlgo.FindMostProbableStateSequence(model, data);
	vector<vector<size_t>> confusionMatrix = estimator.CombineConfusionMatrix(data, mostProbableSeq, model);
	vector<EvaluationMetrics> estimations = estimator.GetStatePredicionEstimations(confusionMatrix);

	cout << "Viterbi algorithm state prediction estimations:" << endl;
	for (size_t i = 0; i < estimations.size(); i++)
		estimator.PrintPredictionsEstimations(i, estimations[i], model);

	cout << "**********************************************************";
	cout << "\n\n";
}

void Test2()
{
	ifstream modelSource("../Model/CustomDefault.model");
	ifstream dataSource("../Dataset/CustomDefault.data");

	if (!modelSource.good())
	{
		cerr << "ERROR: failed to open model file properly. " << endl;
		return;
	}

	if (!dataSource.good())
	{
		cerr << "ERROR: failed to open data file properly. " << endl;
		return;
	}

	// enable exceptions to signal errors later while reading model and data
	ios_base::iostate ioExcept = (ifstream::failbit | ifstream::badbit | ifstream::eofbit);

	modelSource.exceptions(ioExcept);
	dataSource.exceptions(ioExcept);

	// section: read model and data
	HiddenMarkovModel model;
	TestData data;

	try
	{
		model.ReadModel(modelSource);
	}
	catch (const exception & e)
	{
		cerr << "ERROR: Fatal problem while reading model. Details: '" << e.what() << "'" << endl;
		return;
	}

	try
	{
		data.ReadTestData(model, dataSource);
	}
	catch (const exception & e)
	{
		cerr << "ERROR: Fatal problem while reading experiment data. Details: '" << e.what() << "'" << endl;
		return;
	}

	// secton: run and estimate forward-backward predictions
	HHMAlgorithm hhmAlgo;
	Estimator estimator;
	vector<size_t> mostProbableSeq = hhmAlgo.FindMostProbableStateSequence(model, data);

	vector<vector<pair<double, double>>> forwardBackwardProb = hhmAlgo.CalcualteForwardBackwardProbabilites(model, data);
	vector<size_t> mostProbableStates = estimator.GetMostProbableStates(forwardBackwardProb);
	vector<vector<size_t>> confusionMatrix = estimator.CombineConfusionMatrix(data, mostProbableSeq, model);
	vector<EvaluationMetrics> estimations = estimator.GetStatePredicionEstimations(confusionMatrix);

	cout << "Forward-Backward algorithm state prediction estimation:" << endl;
	for (size_t i = 0; i < estimations.size(); i++)
		estimator.PrintPredictionsEstimations(i, estimations[i], model);

	cout << "\n";
}

void Test3()
{
	ifstream modelSource("../Model/conll2000_test.model");

	if (!modelSource.good())
	{
		cerr << "ERROR: failed to open model file properly. " << endl;
		return;
	}

	// enable exceptions to signal errors later while reading model and data
	ios_base::iostate ioExcept = (ifstream::failbit | ifstream::badbit | ifstream::eofbit);

	modelSource.exceptions(ioExcept);

	// section: read model and data
	HiddenMarkovModel model;

	try
	{
		model.ReadModel(modelSource);
	}
	catch (const exception & e)
	{
		cerr << "ERROR: Fatal problem while reading model. Details: '" << e.what() << "'" << endl;
		return;
	}
}

void Test4()
{
	ifstream modelSource("../Model/conll2000_test.model");

	if (!modelSource.good())
	{
		cerr << "ERROR: failed to open model file properly. " << endl;
		return;
	}

	// enable exceptions to signal errors later while reading model and data
	ios_base::iostate ioExcept = (ifstream::failbit | ifstream::badbit | ifstream::eofbit);

	modelSource.exceptions(ioExcept);

	// section: read model and data
	HiddenMarkovModel model;

	try
	{
		model.ReadModel(modelSource);
	}
	catch (const exception & e)
	{
		cerr << "ERROR: Fatal problem while reading model. Details: '" << e.what() << "'" << endl;
		return;
	}

	TestData data;
	data.ReadTestData(model, "../Dataset/conll2000_test.txt", " ");
}


void Test5()
{
	ifstream modelSource("../Model/conll2000_test.model");
	ifstream dataSource("../Dataset/conll2000_SampleTest.data");

	if (!modelSource.good())
	{
		cerr << "ERROR: failed to open model file properly. " << endl;
		return;
	}

	if (!dataSource.good())
	{
		cerr << "ERROR: failed to open data file properly. " << endl;
		return;
	}

	// enable exceptions to signal errors later while reading model and data
	ios_base::iostate ioExcept = (ifstream::failbit | ifstream::badbit | ifstream::eofbit);

	modelSource.exceptions(ioExcept);
	dataSource.exceptions(ioExcept);

	// section: read model and data
	HiddenMarkovModel model;
	TestData data;

	try
	{
		model.ReadModel(modelSource);
	}
	catch (const exception & e)
	{
		cerr << "ERROR: Fatal problem while reading model. Details: '" << e.what() << "'" << endl;
		return;
	}

	try
	{
		data.ReadTestData(model, dataSource);
	}
	catch (const exception & e)
	{
		cerr << "ERROR: Fatal problem while reading experiment data. Details: '" << e.what() << "'" << endl;
		return;
	}

	// secton: run and estimate viterbi predictions
	HHMAlgorithm hhmAlgo;
	Estimator estimator;
	vector<size_t> mostProbableSeq = hhmAlgo.FindMostProbableStateSequence(model, data);
	vector<vector<size_t>> confusionMatrix = estimator.CombineConfusionMatrix(data, mostProbableSeq, model);
	vector<EvaluationMetrics> estimations = estimator.GetStatePredicionEstimations(confusionMatrix);

	cout << "Viterbi algorithm state prediction estimations:" << endl;
	for (size_t i = 0; i < estimations.size(); i++)
		estimator.PrintPredictionsEstimations(i, estimations[i], model);

	cout << "**********************************************************";
	cout << "\n\n";
}

int main()
{
	Test1();
	//Test2();
	//Test3();
	//Test4();
	//Test5();

	return 0;
}