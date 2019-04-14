package to.kishimo.ml4se.ch05;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class LogisticVsPerceptron {
    private static double[] variances = {5.0, 10.0, 15.0, 30.0};

    public static void main(String[] args) {
        for (double variance : variances) {
            LogisticVsPerceptron lp = new LogisticVsPerceptron();
            lp.runSimulation(variance);
        }
    }

    private void runLogistic(List<Point> trainSet) {
        RealMatrix w = MatrixUtils.createRealMatrix(new double[][]{{0.0}, {0.1}, {0.1}});
        RealMatrix phi = createPhi(trainSet);
        RealMatrix t = createT(trainSet);
        for (int i = 0; i < 30; i++) {
            RealMatrix y = MatrixUtils.createRealMatrix(1, phi.getRowDimension());
            for (int j = 0; j < phi.getRowDimension(); j++) {
                RealMatrix line = phi.getRowMatrix(j);
                RealMatrix a = line.multiply(w);
                y.setEntry(0, j, 1.0 / (1.0 + Math.exp(-a.getEntry(0, 0))));
            }
            RealMatrix r = MatrixUtils.createRealDiagonalMatrix(y.getRow(0));
            y = y.transpose();
            RealMatrix tmp1 = MatrixUtils.inverse(phi.transpose().multiply(r).multiply(phi));
            RealMatrix tmp2 = phi.transpose().multiply(y.subtract(t));
            RealMatrix wNew = w.subtract(tmp1.multiply(tmp2));
            if (wNew.subtract(w).transpose().multiply(wNew.subtract(w)).getEntry(0, 0) < 0.001 * w.transpose().multiply(w).getEntry(0, 0)) {
                w = wNew;
                break;
            }
            w = wNew;
        }
        double err = 0.0;
        for (Point p : trainSet) {
            int type = p.type * 2 - 1;
            if (type * (w.getEntry(0, 0) + w.getEntry(1, 0) * p.x + w.getEntry(2, 0) * p.y) < 0) {
                err += 1.0;
            }
        }
        double errRate = err * 100 / trainSet.size();
        System.out.println("LogisticRegression Error Rate: " + errRate + "%");
    }

    private RealMatrix createPhi(List<Point> trainSet) {
        RealMatrix phi = MatrixUtils.createRealMatrix(trainSet.size(), 3);
        for (int i = 0; i < trainSet.size(); i++) {
            phi.setRow(i, new double[]{1.0, trainSet.get(i).x, trainSet.get(i).y});
        }
        return phi;
    }

    private RealMatrix createT(List<Point> trainSet) {
        RealMatrix t = MatrixUtils.createRealMatrix(trainSet.size(), 1);
        for (int i = 0; i < trainSet.size(); i++) {
            t.setEntry(i, 0, trainSet.get(i).type);
        }
        return t;
    }

    private void runPerceptron(List<Point> trainSet) {
        double w0 = 0.0;
        double w1 = 0.0;
        double w2 = 0.0;
        double bias = 0.0;
        for (Point p : trainSet) {
            bias += (Math.abs(p.x) + Math.abs(p.y)) / 2.0;
        }
        bias /= trainSet.size();

        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < trainSet.size(); j++) {
                Point p = trainSet.get(j);
                int type = p.type * 2 - 1;
                if (type * (w0 * bias + w1 * p.x + w2 * p.y) <= 0) {
                    w0 += type * bias;
                    w1 += type * p.x;
                    w2 += type * p.y;
                }
            }
        }

        double err = 0.0;
        for (Point p : trainSet) {
            int type = p.type * 2 - 1;
            if (type * (w0 * bias + w1 * p.x + w2 * p.y) <= 0) {
                err += 1.0;
            }
        }
        double errRate = err * 100 / trainSet.size();
        System.out.println("Perceptron Error Rate: " + errRate + "%");
    }

    private void runSimulation(double variance) {
        List<Point> trainSet = prepareDataset(variance);
        runLogistic(trainSet);
        runPerceptron(trainSet);
    }

    private List<Point> prepareDataset(double var) {
        int n1 = 10;
        int n2 = 10;
        double[] mu1 = {7.0, 7.0};
        double[] mu2 = {-3.0, -3.0};
        List<Point> df1 = variateNormal(mu1, var, 1, n1);
        List<Point> df2 = variateNormal(mu2, var, 0, n2);
        List<Point> df = new ArrayList<>();
        df.addAll(df1);
        df.addAll(df2);
        Collections.shuffle(df);
        return df;
    }

    private List<Point> variateNormal(double[] mu, double var, int type, int num) {
        List<Point> dataset = new ArrayList<>();
        for (int i = 0; i < num; i++) {
            Point p = new Point(mu, var, type);
            dataset.add(p);
        }
        return dataset;
    }

    private static class Point {
        private static Random rand = new Random();

        private double x;
        private double y;
        private int type;

        private Point(double[] mu, double var, int type) {
            this.type = type;
            x = mu[0] + rand.nextGaussian() * Math.sqrt(var);
            y = mu[1] + rand.nextGaussian() * Math.sqrt(var);
        }
    }
}
