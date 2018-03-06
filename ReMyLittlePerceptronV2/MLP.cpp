#include"MLP.h"

//TrainingElement
MyLittlePerceptron::TrainingElement::TrainingElement(vector<float> in, vector<float> out) {
	this->in = in;
	this->out = out;
}

//Layer
MyLittlePerceptron::Layer::Layer(int dimension) {
	dimension = dimension;
	for (int i = 0; i < dimension; ++i) {
		in.push_back(0);
		out.push_back(0);
		err.push_back(0);
	}
}

//WeightMatrix
MyLittlePerceptron::WeightMatrix::WeightMatrix(int inputDimension, int outputDimension)
{
	this->inputDimension = inputDimension;
	this->outputDimension = outputDimension;
	for (int i = 0; i < inputDimension*outputDimension; ++i)
		weights.push_back((float)(rand()/RAND_MAX));
}

//constructors and destroyers
MyLittlePerceptron::MyLittlePerceptron(int inDim, int outDim) {
	inputDimension = inDim;
	outputDimension = outDim;
	layers.push_back(Layer(inputDimension));
	Height = 1;
}
MyLittlePerceptron::~MyLittlePerceptron() {

}

//method
void MyLittlePerceptron::addHiddenLayer(int layerDimension) {
	addLayer(layerDimension);
}
void MyLittlePerceptron::addLayer(int layerDimension) {
	layers.push_back(Layer(layerDimension));
	Height++;
}
void MyLittlePerceptron::resetNetwork() {
	resetWeights();
}
void MyLittlePerceptron::initialization() {
	addLayer(outputDimension);
	resetNetwork();
}
void MyLittlePerceptron::resetWeights() {
	weightMatricies.clear();
	int i;
	for (; i < Height - 1; ++i)
		weightMatricies.push_back(WeightMatrix(layers[i].dimension,layers[i+1].dimension));
}

//currently using softsign
float MyLittlePerceptron::activationFunction(float input) {
	return input / (1 + abs(input));
}
//derivate form of activationFunction
float MyLittlePerceptron::dxActivationFunction(float input) {
	return input / pow((1 + abs(input)),2);
}

void MyLittlePerceptron::calcLayerInput(int h) {
	if (!(h > 0 && h < Height)) return;
	WeightMatrix *wm = &(weightMatricies[h - 1]);
	for (int i = 0; i < layers[h].dimension; ++i) {
		layers[h].in[i] = 0;
		for (int j = 0; j < layers[h - 1].dimension; ++j) {
			layers[h].in[i] += layers[h - 1].in[j] * wm->weights[i*wm->inputDimension + j];
		}
	}
}

void MyLittlePerceptron::calcLayerOutput(int h) {
	for (int i = 0; i < layers[h].dimension; ++i)
		layers[h].out[i] = activationFunction(layers[h].in[i]);
}

vector<float> MyLittlePerceptron::classify(vector<float> in) {
	if (in.size != inputDimension) {
		printf("Error :: %s :: input demension is not equal",__FUNCTION__);
		return;
	}
	for (int i = 0; i < Height; ++i) {
		calcLayerInput(i);
		calcLayerOutput(i);
	}
	return layers[Height - 1].out;
}
