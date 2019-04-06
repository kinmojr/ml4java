package to.kishimo.deeplearning.chap3;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Chap3Main {
    public static void main(String... args) {
        // # 3.4 3層ニューラルネットワークの実装
        double[][] w1 = {{0.1, 0.3, 0.5}, {0.2, 0.4, 0.6}};
        double[] b1 = {0.1, 0.2, 0.3};
        double[][] w2 = {{0.1, 0.4}, {0.2, 0.5}, {0.3, 0.6}};
        double[] b2 = {0.1, 0.2};
        double[][] w3 = {{0.1, 0.3}, {0.2, 0.4}};
        double[] b3 = {0.1, 0.2};
        double[][] x = {{1.0, 0.5}};

        double[][] a1 = addBias(dotProduct(x, w1), b1);
        double[][] z1 = sigmoid(a1);
        double[][] a2 = addBias(dotProduct(z1, w2), b2);
        double[][] z2 = sigmoid(a2);
        double[][] a3 = addBias(dotProduct(z2, w3), b3);

        for (int i = 0; i < a3.length; i++) {
            for (int j = 0; j < a3[0].length; j++) {
                System.out.print(a3[i][j] + ", ");
            }
            System.out.println(a1.length);
        }
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
     * ReLU関数.
     */
    public static double relu(double value) {
        if (value < 0) return 0;
        else return value;
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

    /**
     * 行列の内積.
     */
    public static double[][] dotProduct(double[][] a, double[][] b) {
        RealMatrix aMat = MatrixUtils.createRealMatrix(a);
        RealMatrix bMat = MatrixUtils.createRealMatrix(b);
        return aMat.multiply(bMat).getData();

        //        int rowSize = a.length;
//        int colSize = b[0].length;
//        int size = a[0].length;
//        double[][] ret = new double[rowSize][colSize];
//        for (int i = 0; i < rowSize; i++) {
//            for (int j = 0; j < colSize; j++) {
//                for (int k = 0; k < size; k++) {
//                    ret[i][j] += a[i][k] * b[k][j];
//                }
//            }
//        }
//        return ret;
    }

    public static double[][] addBias(double[][] values, double[] bias) {
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                values[i][j] += bias[j];
            }
        }
        return values;
    }

    /**
     * 重みファイルを読み込んで重みデータを作成する.
     *
     * @param file 重みファイルのパス
     * @return 重みデータ
     */
    public static RealMatrix readWeights(String file) throws IOException {
        System.out.println("Loading weight data from " + file + " ...");
        List<String> lines = new ArrayList<String>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(ClassLoader
                    .getSystemResourceAsStream(file)));

            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
            }
        } finally {
            if (br != null)
                br.close();
        }

        double[][] matrix = new double[lines.size()][lines.get(0).split("\t").length];
        for (int i = 0; i < matrix.length; i++) {
            String[] values = lines.get(i).split("\t");
            for (int j = 0; j < values.length; j++) {
                matrix[i][j] = Double.valueOf(values[j]);
            }
        }

        return MatrixUtils.createRealMatrix(matrix);
    }

    public static int getMaxIndex(double[] values) {
        double maxValue = Double.NaN;
        int maxIndex = 0;
        for (int i = 0; i < values.length; i++) {
            if (Double.isNaN(maxValue)) {
                maxValue = values[i];
            } else if (values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}

