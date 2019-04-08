package to.kishimo.ml4se.chap03;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class MaximumLikelihood {
    private static Random rand = new Random();

    public static void main(String[] args) {
        int N = 10;
        int[] M = new int[]{0, 1, 3, 9};

        MaximumLikelihood ml = new MaximumLikelihood();
        RealMatrix trainSet = ml.createDataset(N);
        RealMatrix testSet = ml.createDataset(N);

        for (int m : M) {
            RealMatrix ws = ml.resolve(trainSet, m);
            double sigma = ml.sigma(trainSet, ws);
            System.out.println("m=" + m + ", Sigma: " + sigma);
        }
        System.out.println();

        for (int m = 0; m < N; m++) {
            RealMatrix ws = ml.resolve(trainSet, m);
            double trainLikelihood = ml.logLikelihood(trainSet, ws);
            double testLikelihood = ml.logLikelihood(testSet, ws);
            System.out.println("m=" + m + ", Train Log Likelihood: " + trainLikelihood + ", Test Log Likelihood: " + testLikelihood);
        }
    }

    private RealMatrix createDataset(int rows) {
        RealMatrix ret = MatrixUtils.createRealMatrix(rows, 2);
        for (int i = 0; i < rows; i++) {
            double x = (double) i / (double) (rows - 1);
            double y = Math.sin(2 * Math.PI * x) + rand.nextGaussian() * 0.3;
            ret.setEntry(i, 0, x);
            ret.setEntry(i, 1, y);
        }
        return ret;
    }

    private double logLikelihood(RealMatrix dataset, RealMatrix ws) {
        double dev = 0.0;
        double n = dataset.getRowDimension();
        for (int i = 0; i < dataset.getRowDimension(); i++) {
            double x = dataset.getEntry(i, 0);
            double y = dataset.getEntry(i, 1);
            dev += Math.pow((y - f(x, ws)), 2.0);
        }
        double err = dev * 0.5;
        double beta = n / dev;
        double lp = -beta * err + 0.5 * n * Math.log(0.5 * beta / Math.PI);
        return lp;
    }

    private double f(double x, RealMatrix ws) {
        double y = 0.0;
        for (int i = 0; i < ws.getRowDimension(); i++) {
            double w = ws.getEntry(i, 0);
            y += w * Math.pow(x, i);
        }
        return y;
    }

    private RealMatrix resolve(RealMatrix dataset, int m) {
        RealMatrix t = dataset.getColumnMatrix(1);
        RealMatrix phi = MatrixUtils.createRealMatrix(dataset.getRowDimension(), m + 1);
        for (int i = 0; i < dataset.getRowDimension(); i++) {
            for (int j = 0; j < m + 1; j++) {
                phi.setEntry(i, j, Math.pow(dataset.getEntry(i, 0), j));
            }
        }
        RealMatrix tmp = MatrixUtils.inverse(phi.transpose().multiply(phi));
        return tmp.multiply(phi.transpose()).multiply(t);
    }

    private double sigma(RealMatrix dataset, RealMatrix ws) {
        double sigma2 = 0.0;
        for (int i = 0; i < dataset.getRowDimension(); i++) {
            double x = dataset.getEntry(i, 0);
            double y = dataset.getEntry(i, 1);
            sigma2 += Math.pow((f(x, ws) - y), 2.0);
        }
        sigma2 /= dataset.getRowDimension();
        return Math.sqrt(sigma2);
    }
}
