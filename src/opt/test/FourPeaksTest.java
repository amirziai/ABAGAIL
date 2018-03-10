package opt.test;

import java.io.IOException;
import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.*;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
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
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    private static final int[] bitstring_sizes = {20, 50, 80, 100};
    private static final int iter = 30;

    private static final int iter_rhc = 200000;
    private static final int iter_sa = 200000;
    private static final int iter_ga = 1000;
    private static final int iter_mimic = 1000;

    private static final double sa_t = 1E11;
    private static final double sa_cooling = 0.95;

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
    
    public static void main(String[] args) throws IOException {
        String[] fields = {"algo", "iter", "bitstring_size", "val", "time"};
        Writer writer = new CSVWriter("output/fourpeaks2.csv", fields);
        writer.open();

        for (int bitstring_index = 0; bitstring_index < bitstring_sizes.length; bitstring_index++) {
            int N = bitstring_sizes[bitstring_index];
            int T = N / 10;

            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

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
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
                FixedIterationTrainer fit = new FixedIterationTrainer(ga, iter_ga);
                output(writer, ef, fit,"ga", i, ga, N);
            }

            for (int i = 0; i < iter; i++) {
                MIMIC mimic = new MIMIC(200, 5, pop);
                FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iter_mimic);
                output(writer, ef, fit,"mimic", i, mimic, N);
            }
        }

        writer.close();
    }
}
