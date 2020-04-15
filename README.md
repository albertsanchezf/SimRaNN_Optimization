# SimRaNN_Optimization

A model of a Neural Network based on a Multi-Layer Perceptron is defined together with a optimization procuedure based on a Random Search Generator. This program uses a dataset CSV file generate with AdaptDatabase program (https://github.com/albertsanchezf/AdaptDatabase).

**Basic configurations**

-*datasetPath*: The location of the Dataset CSV file.

-*savePathOptimization*: The path of the location of the generated configuartions as well as final results.

-*totalBatches*: this value must be the number of lines of the input dataset. If this number is wrongly set, the number of epochs will vary.

**Advanced Configuration**

-*Batch size*: defined in *batchSize*.

-*Number of epochs*: defined in *numEpochs* (The total epochs is the numerator number)

-*Train and test dataset split ratio*: defined in *ratio*.

-*Hyperparameter definition example*: ParameterSpace<Integer> layer1SizeHyperparam = new IntegerParameterSpace(32,2048);

-*Model configuration*: it is defined under *MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()*

-*Termination conditions configuration*: it is defined under *TerminationCondition[] conditions = {*


