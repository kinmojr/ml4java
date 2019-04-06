package to.kishimo.deeplearning.chap4;

import to.kishimo.deeplearning.util.Functions;

import java.util.Random;

public class Gradient {
    double[][] w;

    public Gradient(int row, int col) {
        w = new double[row][col];
        Random rand = new Random();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                w[i][j] = rand.nextGaussian();
            }
        }
    }

    public Gradient(double[][] w) {
        this.w = w;
    }

    public double[][] predict(double[][] x) {
        return Functions.dotProduct(x, w);
    }

    public double[] predict(double[] x) {
        return Functions.dotProduct(x, w[0]);
    }

    public double loss(double[] x, double[] t) {
        double[] z = predict(x);
        double[] y = Functions.softmax(z);
        double loss = Functions.crossEntropyError(y, t);
        return loss;
    }

    public double[] loss(double[][] x, double[] t) {
        double[][] z = predict(x);
        double[][] y = Functions.softmax(z);
        double[] loss = Functions.crossEntropyError(y, t);
        return loss;
    }

    public static void main(String... args) {
        final Gradient simpleNet = new Gradient(new double[][]{{0.47355232, 0.9977393, 0.84668094}, {0.85557411, 0.03563661, 0.69422093}});
        final double[][] x = new double[][]{{0.6, 0.9}};
        // 1*2 dot 2*3 = 1*3
        double[][] p = simpleNet.predict(x);
        for (double tmp : p[0]) System.out.println(tmp);
        System.out.println(Functions.maxIndex(p[0]));

        final double[] t = new double[]{0.0, 0.0, 1.0};
        System.out.println(simpleNet.loss(x, t)[0]);

//        double[][] dw = Functions.numericalGradient(new Function2() {
//            public double[] function(double[][] values) {
//                return simpleNet.loss(x, t);
//            }
//        }, simpleNet.w);
    }
}

