#ifndef MYLITTLEPERCEPTRON
#define MYLITTLEPERCEPTRON
#include<iostream>
#include<vector>
#include<cstdlib>

using namespace std;

class MyLittlePerceptron {
	public:
		struct TrainingElement {
			//constructor
			TrainingElement(vector<float> in, vector<float> out);
			//member
			vector<float> in;
			vector<float> out;
		};
		//constructor, destroyer
		MyLittlePerceptron(int inDim, int outDim);
		~MyLittlePerceptron();
		//method
		void initialization();
		void addHiddenLayer(int layerDimension);
	private:
		struct WeightMatrix {
			int inputDimension;
			int outputDimension;
			vector<float> weights;
			WeightMatrix(int inputDimension, int outputDimension);
		};
		struct Layer {
			//constructor
			Layer(int dimension);
			//member
			int dimension;
			vector<float> in;
			vector<float> out;
			vector<float> err;
		};
		//member
		int inputDimension;
		int outputDimension;
		int layers;
		int Height;
		
		vector<Layer> layers;
		vector<TrainingElement> trainingSet;
		vector<WeightMatrix> weightMatricies;

		//method
		void addLayer(int layerDimension);
		float activationFunction(float in);
		float dxActivationFunction(float in);
		void resetWeights();
		void resetNetwork();
		void calcLayerInput(int h);
		void calcLayerOutput(int h);
		vector<float> classify(vector<float> in);
};

#endif