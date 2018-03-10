package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.writer.CSVWriter;
import shared.writer.Writer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class StudentTest {
    private static int rows = 649;
    private static int features = 73;

    private static Instance[] instances = initializeInstances();

    private static int inputLayer = features, outputLayer = 1, trainingIterations =
            100;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"GA", "SA", "RHC"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) throws IOException {
        int iter_range = 2000;
        int[] units = {100};
        int[] iters = new int[iter_range];
        for (int i = 0; i < iter_range; i++) {
            iters[i] = i + 1;
        }

         String[] fields = {"units", "iter", "algo", "acc", "time"};
         Writer writer = new CSVWriter("output/nn-v2.csv", fields);
         writer.open();

        for (int u = 0; u < units.length; u++) {
            int unit = units[u];

            // for (int it = 0; it < iters.length; it++) {

            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                        new int[] {inputLayer, unit, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
            }

            oa[0] = new RandomizedHillClimbing(nnop[0]);
            oa[1] = new SimulatedAnnealing(1E0, .95, nnop[1]);
            oa[2] = new StandardGeneticAlgorithm(100, 100, 20, nnop[2]);

            for(int i = 0; i < oa.length; i++) {
                train(oa[i], networks[i], oaNames[i], iter_range, writer); //trainer.train();
            }
        }

        writer.close();
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String
            oaName, int iter, Writer writer) throws IOException {

        for(int i = 0; i < iter; i++) {
            double start = System.nanoTime(), trainingTime;
            oa.train();
            trainingTime = System.nanoTime() - start;
            trainingTime /= Math.pow(10,9);

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            Instance optimalInstance = oa.getOptimal();
            network.setWeights(optimalInstance.getData());

            // System.out.println(df.format(error));
            double correct = 0, incorrect = 0;
            double predicted, actual;
            for (int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(network.getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }

            double acc = correct / (correct + incorrect);
            String out = 100 + "," + (i + 1) + "," + oaName + "," + acc + "," + trainingTime;

            System.out.println(out);
            writer.write(out);
            writer.nextRecord();
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[rows][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/student" +
                    ".csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[features]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < features; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
