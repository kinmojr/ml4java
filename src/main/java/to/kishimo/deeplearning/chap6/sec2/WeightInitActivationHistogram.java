package to.kishimo.deeplearning.chap6.sec2;

import to.kishimo.deeplearning.util.Functions;

import java.util.Random;

public class WeightInitActivationHistogram {
    public static void main(String... args) {
        double[][] inputData = randomMatrix(1000, 100);
        int nodeNum = 100;
        int hiddenLayerSize = 5;
        double[][][] activations = new double[hiddenLayerSize][inputData.length][inputData[0].length];

        double[][] x = copyMatrix(inputData);

        for (int i = 0; i < hiddenLayerSize; i++) {
            if (i != 0)
                x = copyMatrix(activations[i - 1]);

            double[][] w = randomMatrix(nodeNum, nodeNum);
            // double[][] w = randomMatrix(nodeNum, nodeNum, 0.01);
            // double[][] w = randomMatrix(nodeNum, nodeNum, Math.sqrt(1.0 / nodeNum));
            // double[][] w = randomMatrix(nodeNum, nodeNum, Math.sqrt(2.0 / nodeNum));

            double[][] a = Functions.dotProduct(x, w);

            double[][] z = sigmoid(a);
            // double[][] z = reLU(a);
            // double[][] z = tanh(a);
            System.out.println();
        }
    }

    private static double[][] sigmoid(double[][] x) {
        double[][] ret = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                ret[i][j] = 1 / (1 + Math.exp(-x[i][j]));
            }
        }
        return ret;
    }

    private static double[][] reLU(double[][] x) {
        double[][] ret = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                if (x[i][j] > 0) {
                    ret[i][j] = x[i][j];
                } else {
                    ret[i][j] = 0;
                }
                ret[i][j] = 1 / (1 + Math.exp(-x[i][j]));
            }
        }
        return ret;
    }

    private static double[][] tanh(double[][] x) {
        double[][] ret = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                ret[i][j] = Math.tanh(x[i][j]);
            }
        }
        return ret;
    }

    private static double[][] randomMatrix(int rowNum, int colNum) {
        return randomMatrix(rowNum, colNum, 1.0);
    }

    private static double[][] randomMatrix(int rowNum, int colNum, double mult) {
        double[][] ret = new double[rowNum][colNum];
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                ret[i][j] = new Random().nextGaussian();
            }
        }
        return ret;
    }

    private static double[][] copyMatrix(double[][] matrix) {
        double[][] ret = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            System.arraycopy(matrix[i], 0, ret[i], 0, matrix[i].length);
        }
        return ret;
    }
}
