package to.kishimo.deeplearning.util;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import to.kishimo.deeplearning.chap4.Function;

import java.lang.*;
import java.lang.Math;

/**
 * 活性化関数等を定義するクラス.
 */
public class Functions {
    /**
     * ReLU関数.
     */
    public static double relu(double value) {
        if (value < 0) return 0;
        else return value;
    }

    /**
     * ReLU関数の配列版.
     */
    public static double[] relu(double[] values) {
        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            ret[i] = relu(values[i]);
        }
        return ret;
    }

    /**
     * シグモイド関数.
     */
    public static double sigmoid(double value) {
        return 1.0 / (1.0 + java.lang.Math.exp(-value));
    }

    /**
     * シグモイド関数の配列版.
     */
    public static double[] sigmoid(double[] values) {
        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            ret[i] = sigmoid(values[i]);
        }
        return ret;
    }

    /**
     * シグモイド関数の2次元配列版.
     */
    public static double[][] sigmoid(double[][] values) {
        double[][] ret = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            ret[i] = sigmoid(values[i]);
        }
        return ret;
    }

    /**
     * シグモイド関数のRealMatrix版.
     */
    public static RealMatrix sigmoid(RealMatrix matrix) {
        double[][] ret = sigmoid(matrix.getData());
        return MatrixUtils.createRealMatrix(ret);
    }

    /**
     * ステップ関数.
     */
    public static double step(double value) {
        if (value > 0.0) return 1.0;
        else return 0.0;
    }

    /**
     * ステップ関数の配列版.
     */
    public static double[] step(double[] values) {
        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            ret[i] = step(values[i]);
        }
        return ret;
    }

    /**
     * ソフトマックス関数.
     */
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

    /**
     * ソフトマックス関数の配列版.
     */
    public static double[][] softmax(double[][] values) {
        double[][] ret = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            ret[i] = softmax(values[i]);
        }
        return ret;
    }

    public static double crossEntropyError(double[] y, double[] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += t[i] * Math.log(y[i] + Math.pow(10.0, -7.0));
        }
        return -sum;
    }

    public static double[] crossEntropyError(double[][] y, double[] t) {
        double[] ret = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            ret[i] = crossEntropyError(y[i], t);
        }
        return ret;
    }

    public static double[] crossEntropyError(double[][] y, double[][] t) {
        double[] ret = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            ret[i] = crossEntropyError(y[i], t[i]);
        }
        return ret;
    }

    public static double numericalDiff(Function func, double[] values) {
        double diff = Math.pow(10, -4);

        double[] values1 = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            values1[i] = values[i] + diff;
        }

        double[] values2 = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            values2[i] = values[i] - diff;
        }

        return (func.function(values1) - func.function(values2)) / (2 * diff);
    }

    public static double[] numericalGradient(Function func, double[] values) {
        double diff = Math.pow(10, -4);
        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            double tmp = values[i];

            values[i] = tmp + diff;
            double value1 = func.function(values);

            values[i] = tmp - diff;
            double value2 = func.function(values);

            ret[i] = (value1 - value2) / (2 * diff);
            values[i] = tmp;
        }
        return ret;
    }

//    public static double[][] numericalGradient(Function2 func, double[][] values) {
//        double diff = Math.pow(10, -4);
//        double[][] ret = new double[values.length][values[0].length];
//        for (int i = 0; i < values.length; i++) {
//            ret[i] = numericalGradient(func, values[i]);
//        }
//        return ret;
//    }

    public static void main(String... args) {
//        double[] t = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
//        double[] y = {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0};
//        System.out.println(crossEntropyError(y, t));

//        y=0.01x^2+0.1xの微分
//        double value = numericalDiff(new Function() {
//            public double function(double[] values) {
//                return 0.01 * Math.pow(values[0], 2.0) + 0.1 * values[0];
//            }
//        }, new double[]{10.0});
//        System.out.println(value);

        // f(x0, x1)=x0^2+x1^2、x0=3、x1=4、x0に対する偏微分
//        double value = numericalDiff(new Function() {
//            public double function(double[] values) {
//                return Math.pow(values[0], 2.0) + Math.pow(4, 2.0);
//            }
//        }, new double[]{3, 4});
//        System.out.println(value);

        // f(x0, x1)=x0^2+x1^2、x0=3、x1=4、x1に対する偏微分
//        double value = numericalDiff(new Function() {
//            public double function(double[] values) {
//                return Math.pow(3, 2.0) + Math.pow(values[1], 2.0);
//            }
//        }, new double[]{3, 4});
//        System.out.println(value);

//        // f(x0, x1)=x0^2+x1^2、x0=3、x1=4の勾配
//        double[] values = numericalGradient(new Function2ValsPow(), new double[]{3, 4});
//        for (double value : values) {
//            System.out.println(value);
//        }

//        RealMatrix a = MatrixUtils.createRealMatrix(new double[][]{{1, 2, 3}, {3, 4, 5}});
//        RealMatrix b = MatrixUtils.createRealMatrix(new double[][]{{5, 6}, {7, 8}, {9, 10}});
//        RealMatrix c = a.multiply(b);
//        System.out.println(c);
//
//        double[][] d = dotProduct(new double[][]{{1, 2, 3}, {3, 4, 5}}, new double[][]{{5, 6}, {7, 8}, {9, 10}});
//        for (double[] e : d) {
//            for (double f : e) {
//                System.out.print(f + ", ");
//            }
//            System.out.println();
//        }

        double[] values = gradientDescent(new Function2ValsPow(), new double[]{-3.0, 4.0}, 0.1, 100);
        for (double value : values) System.out.println(value);
    }

    public static double[] gradientDescent(Function func, double[] initX, double lr, int stepNum) {
        double[] x = initX;
        for (int i = 0; i < stepNum; i++) {
            double[] grad = numericalGradient(new Function2ValsPow(), x);
            for (int j = 0; j < x.length; j++) {
                x[j] -= lr * grad[j];
            }
        }
        return x;
    }

    static class Function2ValsPow implements Function {
        public double function(double[] values) {
            return Math.pow(values[0], 2.0) + Math.pow(values[1], 2.0);
        }
    }

    public static double[][] dotProduct(double[][] a, double[][] b) {
        int rowSize = a.length;
        int colSize = b[0].length;
        int size = a[0].length;
        double[][] ret = new double[rowSize][colSize];
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                for (int k = 0; k < size; k++) {
                    ret[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return ret;
    }

    public static double[] dotProduct(double[] a, double[] b) {
        int size = a.length;
        double[] ret = new double[size];
        for (int i = 0; i < size; i++) {
            ret[i] = a[i] * b[i];
        }
        return ret;
    }

    /**
     * 引数の配列の中から最大値を返す.
     */
    public static double max(double[] values) {
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

    public static int maxIndex(double[] values) {
        double max = Double.NaN;
        int index = 0;
        for (int i = 0; i < values.length; i++) {
            if (Double.isNaN(max)) {
                max = values[i];
            } else if (values[i] > max) {
                max = values[i];
                index = i;
            }
        }
        return index;
    }
}
