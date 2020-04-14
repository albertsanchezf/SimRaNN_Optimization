package org.asanchezf.SimRaNN_Optimization;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.impl.LoggingStatusListener;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

public class Classifier {

    static org.apache.log4j.Logger log = Logger.getLogger(Classifier.class);
    private static String datasetPath = "/Users/AlbertSanchez/Desktop/TFM (noDropBox)/Dataset/binaryDS/dataset.csv";
    private static String savePathOptimization = "resources/optimization/binaryDS/";
    private static int labelIndex = 28;  //15 values in each row of the dataset.csv; CSV: 14 input features followed by an integer label (class) index. Labels are the 15th value (index 14) in each row
    private static int numClasses = 1;   //0 - No incident | 1 - Incident
    private static int totalBatches = 3896; //SimRa binary dataset: 3896
    private static int batchSize = 512;
    private static double ratio = 0.7;
    private static int numEpoch = 1000/((int)Math.ceil((double)totalBatches/(double)batchSize)); //We set the total iteration to 1000. For these reason the nEpochs depend on the totalBatches and the batchSize

    public static void main(String[] args) throws  Exception {

        String configFilename = System.getProperty("user.dir")
                + File.separator + "log4j.properties";
        PropertyConfigurator.configure(configFilename);


        ParameterSpace<Integer> layer1SizeHyperparam = new IntegerParameterSpace(32,2048); //Integer values will be
        ParameterSpace<Integer> layer2SizeHyperparam = new IntegerParameterSpace(32,2048); //generated uniformly at random
        ParameterSpace<Integer> layer3SizeHyperparam = new IntegerParameterSpace(32,2048); //between 16 and 256 (inclusive)
        ParameterSpace<Integer> layer4SizeHyperparam = new IntegerParameterSpace(32,2048); //between 16 and 256 (inclusive)
        ParameterSpace<Integer> layer5SizeHyperparam = new IntegerParameterSpace(32,2048); //between 16 and 256 (inclusive)

        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(1e-4)
                .updater(new Sgd(0.1))
                .addLayer(new DenseLayerSpace.Builder()
                    .nIn(labelIndex)
                    .nOut(layer1SizeHyperparam)
                    .activation(Activation.RELU)
                    .build())
                .addLayer(new DenseLayerSpace.Builder()
                    .nIn(layer1SizeHyperparam)
                    .nOut(layer2SizeHyperparam)
                    .activation(Activation.RELU)
                    .build())
                .addLayer(new DenseLayerSpace.Builder()
                    .nIn(layer2SizeHyperparam)
                    .nOut(layer3SizeHyperparam)
                    .activation(Activation.RELU)
                    .build())
                .addLayer(new DenseLayerSpace.Builder()
                    .nIn(layer3SizeHyperparam)
                    .nOut(layer4SizeHyperparam)
                    .activation(Activation.RELU)
                    .build())
                .addLayer(new DenseLayerSpace.Builder()
                    .nIn(layer4SizeHyperparam)
                    .nOut(layer5SizeHyperparam)
                    .activation(Activation.RELU)
                    .build())
                .addLayer(new OutputLayerSpace.Builder()
                    .nIn(layer5SizeHyperparam)
                    .nOut(numClasses)
                    .activation(Activation.SIGMOID)
                    .lossFunction(LossFunctions.LossFunction.XENT)
                    .build())
                .numEpochs(numEpoch)
                .build();


        // (a) How are we going to generate candidates? (random search or grid search)
        CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace,null);

        // (b) How are going to provide data?
        Class<? extends DataSource> dataSourceClass = SimRaDataSource.class;
        Properties dataSourceProperties = new Properties();
        dataSourceProperties.setProperty("minibatchSize", String.valueOf(batchSize));

        // (c) How we are going to save the models that are generated and tested?
        File f = new File(savePathOptimization);
        if (f.exists()) f.delete();
        f.mkdir();
        ResultSaver modelSaver = new FileModelSaver(savePathOptimization);

        // (d) What are we actually trying to optimize?
        ScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.ACCURACY);

        // (e) When should we stop searching? Specify this with termination conditions
        TerminationCondition[] conditions = {
                new MaxTimeCondition(120, TimeUnit.MINUTES),
                new MaxCandidatesCondition(10)
        };

        //Given these configuration options, let's put them all together:
        OptimizationConfiguration optimizationConfiguration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .dataSource(dataSourceClass,dataSourceProperties)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(conditions)
                .build();

        //And set up execution locally on this machine:
        IOptimizationRunner runner = new LocalOptimizationRunner(optimizationConfiguration,new MultiLayerNetworkTaskCreator());

        //Uncomment this if you want to store the model.
        // StatsStorage ss = new FileStatsStorage(new File("HyperParamOptimizationStats.dl4j"));
        runner.addListeners(new LoggingStatusListener()); //new ArbiterStatusListener(ss)

        //Start the hyperparameter optimization
        runner.execute();

        //Print the best hyper params

        double bestScore = runner.bestScore();
        int bestCandidateIndex = runner.bestScoreCandidateIndex();
        int numberOfConfigsEvaluated = runner.numCandidatesCompleted();

        String s = "Best score: " + bestScore + "\n" +
                "Index of model with best score: " + bestCandidateIndex + "\n" +
                "Number of configurations evaluated: " + numberOfConfigsEvaluated + "\n";

        System.out.println(s);

        //Get all results, and print out details of the best result:
        List<ResultReference> allResults = runner.getResults();

        OptimizationResult bestResult = allResults.get(bestCandidateIndex).getResult();
        MultiLayerNetwork bestModel = (MultiLayerNetwork) bestResult.getResultReference().getResultModel();

        String bestModelJSON = bestModel.getLayerWiseConfigurations().toJson();
        String bestCandidatePath = System.getProperty("user.dir") + "/" +savePathOptimization + String.valueOf(bestCandidateIndex);

        try{saveJSONConfig(bestModelJSON,bestCandidatePath);}
        catch(Exception e){System.out.println("Error saving best config json. More information: " + e.getMessage());}

        try{removeFiles(bestCandidateIndex);}
        catch(Exception e){System.out.println("Error trying to remove no optimal configuration folders. More information: " + e.getMessage());}


    }

    public static void saveJSONConfig(String json, String folder) throws IOException
    {
        PrintWriter writer = new PrintWriter(folder + "/modelConfig.json", "UTF-8");
        writer.println(json);
        writer.close();
    }

    public static void removeFiles(int bestCandidateIndex) throws IOException {
        File removeFolder;
        File resultsFolder = new File(System.getProperty("user.dir") + "/" + savePathOptimization);
        for (String folder : resultsFolder.list())
        {
            if (!folder.startsWith("."))
            {
                if (Integer.valueOf(folder) != bestCandidateIndex)
                {
                    removeFolder = new File(System.getProperty("user.dir") + "/" + savePathOptimization + folder);
                    FileUtils.deleteDirectory(removeFolder);
                }
            }
        }
    }

    public static class SimRaDataSource implements DataSource{


        private int minibatchSize;

        public SimRaDataSource(){

        }

        @Override
        public void configure(Properties properties) {
            this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize", String.valueOf(batchSize)));
        }

        @Override
        public Object trainData() {
            try{
                DataSetIterator iterator = new RecordReaderDataSetIterator(dataPreprocess(),minibatchSize,labelIndex,numClasses);
                return dataSplit(iterator).getTrainIterator();
            }
            catch(Exception e){
                throw new RuntimeException();
            }
        }

        @Override
        public Object testData() {
            try{
                DataSetIterator iterator = new RecordReaderDataSetIterator(dataPreprocess(),minibatchSize,labelIndex,numClasses);
                return dataSplit(iterator).getTestIterator();
            }
            catch(Exception e){
                throw new RuntimeException();
            }
        }

        @Override
        public Class<?> getDataType() {
            return DataSetIterator.class;
        }

        public DataSetIteratorSplitter dataSplit(DataSetIterator iterator) throws IOException, InterruptedException {
            DataNormalization dataNormalization = new NormalizerStandardize();
            dataNormalization.fit(iterator);
            iterator.setPreProcessor(dataNormalization);
            DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator, totalBatches, ratio);
            return splitter;
        }

        public RecordReader dataPreprocess() throws IOException, InterruptedException {
            //Schema Definitions
            Schema s = new Schema.Builder()
                    .addColumnsFloat("speed","mean_acc_x","mean_acc_y","mean_acc_z","std_acc_x","std_acc_y","std_acc_z")
                    .addColumnDouble("sma")
                    .addColumnFloat("mean_svm")
                    .addColumnsDouble("entropyX","entropyY","entropyZ")
                    .addColumnsInteger("bike_type","phone_location","incident_type")
                    .build();

            //Schema Transformation
            TransformProcess tp = new TransformProcess.Builder(s)
                    .integerToOneHot("bike_type",0,8)
                    .integerToOneHot("phone_location",0,6)
                    //.removeColumns("speed")
                    //.removeColumns("mean_acc_x")
                    //.removeColumns("mean_acc_y")
                    //.removeColumns("mean_acc_z")
                    //.removeColumns("std_acc_x")
                    //.removeColumns("std_acc_y")
                    //.removeColumns("std_acc_z")
                    //.removeColumns("sma")
                    //.removeColumns("mean_svm")
                    //.removeColumns("entropyX")
                    //.removeColumns("entropyY")
                    //.removeColumns("entropyZ")
                    //.removeColumns("bike_type")
                    //.removeColumns("phone_location")
                    //.removeColumns("incident_type")
                    .build();

            // Set the number of inputs (labelIndex) depending on the TransformProcess
            labelIndex = tp.getFinalSchema().getColumnNames().size() - 1;

            //CSVReader - Reading from file and applying transformation
            RecordReader rr = new CSVRecordReader(0,',');
            rr.initialize(new FileSplit(new File(datasetPath)));
            RecordReader transformProcessRecordReader = new TransformProcessRecordReader(rr,tp);

            return transformProcessRecordReader;
        }
    }

}

