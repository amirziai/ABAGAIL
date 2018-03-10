package opt.test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.*;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.writer.CSVWriter;
import shared.writer.Writer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    // private static final int N = 50;
    private static final int[] bitstring_sizes = {50};
    private static final int iter_rhc = 200000;
    private static final int iter_sa = 200000;
    private static final int iter_ga = 1000;
    private static final int iter_mimic = 1000;

    private static final double sa_t = 1E12;
    private static final double sa_cooling = 0.95;

    private static final int iter = 1;


    private static void output(Writer writer, EvaluationFunction ef, FixedIterationTrainer fit,
                               String algoName, int iter, OptimizationAlgorithm algo,
                               int bitstring_size)
            throws IOException {
        long t = System.nanoTime();
        fit.train();
        double time = ((double)(System.nanoTime() - t))/ 1e9d;
        double optimal = ef.value(algo.getOptimal());
        String out = algoName + "," + (iter + 1) + "," + bitstring_size + "," + optimal + "," +
                time;
        writer.write(out);
        writer.nextRecord();
        System.out.println(out);
    }

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws IOException {
        Random random = new Random();

        String[] fields = {"algo", "iter", "bitstring_size", "val", "time"};
        Writer writer = new CSVWriter("output/tsm-2.csv", fields);
        writer.open();

        for (int bitstring_index = 0; bitstring_index < bitstring_sizes.length; bitstring_index++) {
            int N = bitstring_sizes[bitstring_index];

            // create the random points
            double[][] points = new double[N][2];
            for (int i = 0; i < points.length; i++) {
                points[i][0] = random.nextDouble();
                points[i][1] = random.nextDouble();
            }
            // for rhc, sa, and ga we use a permutation based encoding
            TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
            Distribution odd = new DiscretePermutationDistribution(N);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            for (int i = 0; i < iter; i++) {
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter_rhc);
                output(writer, ef, fit,"rhc", i, rhc, N);
            }

            for (int i = 0; i < iter; i++) {
                SimulatedAnnealing sa = new SimulatedAnnealing(sa_t, sa_cooling, hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(sa, iter_sa);
                output(writer, ef, fit,"sa", i, sa, N);
            }

            for (int i = 0; i < iter; i++) {
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
                FixedIterationTrainer fit = new FixedIterationTrainer(ga, iter_ga);
                output(writer, ef, fit,"ga", i, ga, N);
            }

            for (int i = 0; i < iter; i++) {
                // for mimic we use a sort encoding
                ef = new TravelingSalesmanSortEvaluationFunction(points);
                int[] ranges = new int[N];
                Arrays.fill(ranges, N);
                odd = new  DiscreteUniformDistribution(ranges);
                Distribution df = new DiscreteDependencyTree(.1, ranges);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

                MIMIC mimic = new MIMIC(200, 100, pop);
                FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iter_mimic);
                output(writer, ef, fit,"mimic", i, mimic, N);
            }
        }

        writer.close();
    }
}
