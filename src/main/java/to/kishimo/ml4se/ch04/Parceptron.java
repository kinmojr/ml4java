package to.kishimo.ml4se.ch04;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Parceptron {
    private int n1 = 20;
    private double[] mu1 = {15.0, 10.0};

    private int n2 = 30;
    private double[] mu2 = {0.0, 0.0};

    private static double[] variances = {15.0, 30.0};

    public static void main(String[] args) {
        for (double variance : variances) {
            Parceptron p = new Parceptron();
            p.runSimulation(variance);
        }
    }

    private void runSimulation(double variance) {
        List<Data> trainSet = prepareDataset(variance);

    }

    private List<Data> prepareDataset(double variance) {
        double[][] cov1 = new double[][]{{variance, 0}, {0, variance}};
        double[][] cov2 = new double[][]{{0, variance}, {variance, 0}};
        List<Data> df1 = multivariateNormal(mu1, cov1, 1, n1);
        List<Data> df2 = multivariateNormal(mu2, cov2, -1, n2);
        List<Data> df = new ArrayList<>();
        df.addAll(df1);
        df.addAll(df2);
        Collections.shuffle(df);
        return df;
    }

    private List<Data> multivariateNormal(double[] mu, double[][] cov, int type, int num) {
        List<Data> dataset = new ArrayList<Data>();
        for (int i = 0; i < num; i++) {
            Data data = new Data(mu, cov, type);
            dataset.add(data);
        }
        return dataset;
    }
}
