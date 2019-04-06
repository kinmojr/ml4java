package common;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Compat {
    public static double[] arrange(double start, double end, double step) {
        int size = (int) ((end - start) / step);
        double[] ret = new double[size];
        for (int i = 0; i < size; i++) {
            ret[i] = start + step * i;
        }
        return ret;
    }

    public static double[] sin(double[] values) {
        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            ret[i] = Math.sin(values[i]);
        }
        return ret;
    }

    public static double[] cos(double[] values) {
        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            ret[i] = Math.cos(values[i]);
        }
        return ret;
    }

    public static RealMatrix dot(RealMatrix x, RealMatrix y) {
        return x.multiply(y);
    }

    public static double dot(double[] x, double[] w) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += w[i] * x[i];
        }
        return sum;
    }

    public static RealMatrix readWeights(String file) throws IOException {
        List<String> lines = new ArrayList<String>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(ClassLoader.getSystemResourceAsStream(file)));

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

    public static RealMatrix readBatchedBias(String file, int numRow) throws IOException {
        String line;
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(ClassLoader.getSystemResourceAsStream(file)));
            line = br.readLine();
        } finally {
            if (br != null)
                br.close();
        }

        double[][] matrix = new double[numRow][line.split("\t").length];
        for (int i = 0; i < numRow; i++) {
            String[] values = line.split("\t");
            for (int j = 0; j < values.length; j++) {
                matrix[i][j] = Double.valueOf(values[j]);
            }
        }

        return MatrixUtils.createRealMatrix(matrix);
    }

    public static int argmax(double[] values) {
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

    public static int[] argmax(double[][] values) {
        int[] maxIndices = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            maxIndices[i] = argmax(values[i]);
        }
        return maxIndices;
    }

    public static double[][][] meshgrid(double[] valsA, double[] valsB) {
        double[][][] ret = new double[2][valsB.length][valsA.length];
        double[][] retA = new double[valsB.length][valsA.length];
        for (int i = 0; i < valsB.length; i++) {
            for (int j = 0; j < valsA.length; j++) {
                retA[i][j] = valsA[j];
            }
        }
        double[][] retB = new double[valsB.length][valsA.length];
        for (int i = 0; i < valsB.length; i++) {
            for (int j = 0; j < valsA.length; j++) {
                retB[i][j] = valsB[i];
            }
        }
        return new double[][][]{retA, retB};
    }

    public static double[] flatten(double[][] values) {
        double[] ret = new double[values.length * values[0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                ret[i * values[0].length + j] = values[i][j];
            }
        }
        return ret;
    }

    public static void print(RealMatrix value) {
        print(value.getData());
    }

    public static void print(double[][] value) {
        for (int i = 0; i < value.length; i++) {
            for (int j = 0; j < value[0].length; j++) {
                System.out.print(value[i][j] + ", ");
            }
            System.out.println();
        }
    }

    public static double[][] sum(double[][] value) {
        double[][] ret = new double[1][value[0].length];
        for (int i = 0; i < value.length; i++) {
            for (int j = 0; j < value[0].length; j++) {
                ret[0][j] += value[i][j];
            }
        }
        return ret;
    }
}
