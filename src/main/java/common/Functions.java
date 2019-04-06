package common;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class Functions {
    public static RealMatrix sigmoid(RealMatrix matrix) {
        double[][] ret = sigmoid(matrix.getData());
        return MatrixUtils.createRealMatrix(ret);
    }

    public static double[][] sigmoid(double[][] values) {
        double[][] ret = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            ret[i] = sigmoid(values[i]);
        }
        return ret;
    }

    public static double[] sigmoid(double[] values) {
        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            ret[i] = sigmoid(values[i]);
        }
        return ret;
    }

    public static double sigmoid(double value) {
        return 1.0 / (1.0 + java.lang.Math.exp(-value));
    }

    public static double[] softmax(double[] values) {
        double max = max(values);

        double sum = 0.0;
        for (double value : values) {
            sum += java.lang.Math.exp(value - max);
        }

        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            ret[i] = Math.exp(values[i] - max) / sum;
        }
        return ret;
    }

    public static double[][] softmax(double[][] values) {
        double[][] ret = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            ret[i] = softmax(values[i]);
        }
        return ret;
    }

    public static double[][] softmax(RealMatrix values) {
        double[][] ret = new double[values.getRowDimension()][values.getColumnDimension()];
        double[][] ary = values.getData();
        for (int i = 0; i < ary.length; i++) {
            ret[i] = softmax(ary[i]);
        }
        return ret;
    }

    public static double[][] dot(double[][] a, double[][] b) {
        RealMatrix aMat = MatrixUtils.createRealMatrix(a);
        RealMatrix bMat = MatrixUtils.createRealMatrix(b);
        return aMat.multiply(bMat).getData();
    }

    public static double[][] randnArray(int row, int col) {
        return randnArray(row, col, 1.0);
    }

    public static double[][] randnArray(int row, int col, double coef) {
        Random rand = new Random();
        double[][] ret = new double[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                ret[i][j] = rand.nextGaussian() * coef;
            }
        }
        return ret;
    }

    public static RealMatrix randn(int row, int col) {
        return randn(row, col, 1.0);
    }

    public static RealMatrix randn(int row, int col, double coef) {
        double[][] ret = randnArray(row, col);
        return MatrixUtils.createRealMatrix(ret);
    }

    public static double crossEntropyError(double[] y, double[] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += t[i] * Math.log(y[i] + Math.pow(10.0, -7.0));
        }
        return -sum;
    }

    public static double crossEntropyError(double[][] y, double[][] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum = crossEntropyError(y[i], t[i]);
        }
        return sum / y.length;
    }

    private static double max(double[] values) {
        double max = Double.NaN;
        for (double value : values) {
            if (Double.isNaN(max)) {
                max = value;
            } else if (value > max) {
                max = value;
            }
        }
        return max;
    }
}
