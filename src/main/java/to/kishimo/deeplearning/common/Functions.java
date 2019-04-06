package to.kishimo.deeplearning.common;

import java.util.Random;

public class Functions {
    public static double[][] randomMatrix(int rowNum, int colNum, double scale) {
        double[][] ret = new double[rowNum][colNum];
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                ret[i][j] = new Random().nextGaussian() * scale;
            }
        }
        return ret;
    }

    public static double[][] addBias(double[][] values, double[][] bias) {
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                values[i][j] += bias[0][j];
            }
        }
        return values;
    }

    public static double l2norm(double[][] x) {
        double ret = 0.0;
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                ret += Math.pow(x[i][j], 2);
            }
        }
        return ret;
    }

    public static double[][] argMax(double[][] y) {
        double[][] ret = new double[y.length][y[0].length];
        for (int i = 0; i < y.length; i++) {
            double maxVal = Double.MIN_VALUE;
            int maxIdx = 0;
            for (int j = 0; j < y[0].length; j++) {
                if (y[i][j] > maxVal) {
                    maxVal = y[i][j];
                    maxIdx = j;
                }
            }
            for (int j = 0; j < y[0].length; j++) {
                if (j == maxIdx) {
                    ret[i][j] = 1.0;
                } else {
                    ret[i][j] = 0.0;
                }
            }
        }
        return ret;
    }

    public static double[][] makeMatrix(int rowNum, int colNum, double initVal) {
        double[][] ret = new double[rowNum][colNum];
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                ret[i][j] = initVal;
            }
        }
        return ret;
    }

    public static double[][] calcWeights(double[][] dW, double[][] W, double weightDecayLambda) {
        double[][] ret = new double[dW.length][dW[0].length];
        for (int i = 0; i < dW.length; i++) {
            for (int j = 0; j < dW[0].length; j++) {
                ret[i][j] = dW[i][j] + weightDecayLambda * W[i][j];
            }
        }
        return ret;
    }
}

