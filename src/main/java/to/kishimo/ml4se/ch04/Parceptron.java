package to.kishimo.ml4se.ch04;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

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
        List<Point> trainSet = prepareDataset(variance);
        double w0 = 0.0;
        double w1 = 0.0;
        double w2 = 0.0;
        double bias = 0.0;
        for (Point p : trainSet) {
            bias += (Math.abs(p.x) + Math.abs(p.y)) / 2.0;
        }
        bias /= trainSet.size();

        double[][] paramhist = new double[30][3];
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < trainSet.size(); j++) {
                Point p = trainSet.get(j);
                if (p.type * (w0 * bias + w1 * p.x + w2 * p.y) <= 0) {
                    w0 += p.type * bias;
                    w1 += p.type * p.x;
                    w2 += p.type * p.y;
                }
            }
            paramhist[i][0] = w0;
            paramhist[i][1] = w1;
            paramhist[i][2] = w2;
        }

        double err = 0.0;
        for (Point p : trainSet) {
            if (p.type * (w0 * bias + w1 * p.x + w2 * p.y) <= 0) {
                err += 1.0;
            }
        }
        double errRate = err * 100 / trainSet.size();
        System.out.println("Error Rate: " + errRate + "%");
        for (int i = 0; i < paramhist.length; i++) {
            System.out.println(i + 1 + " w0: " + paramhist[i][0] + ", w1: " + paramhist[i][1] + ", w2: " + paramhist[i][2]);
        }
        System.out.println();
    }

    private List<Point> prepareDataset(double var) {
        List<Point> df1 = variateNormal(mu1, var, 1, n1);
        List<Point> df2 = variateNormal(mu2, var, -1, n2);
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
