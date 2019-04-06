package to.kishimo.ml4se.chap02;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class SquareError {
    private static Random rand = new Random();
    private int N;
    private int[] M;

    public SquareError(int N, int[] M) {
        this.N = N;
        this.M = M;
    }

    private double[][] createDataset(int num) {
        double[][] ret = new double[num][2];
        for (int i = 0; i < num; i++) {
            double x = (double) i / (double) (num - 1);
            double y = Math.sin(2 * Math.PI * x) + rand.nextGaussian() * 0.3;
            ret[i][0] = x;
            ret[i][1] = y;
        }
        return ret;
    }

    private double rmsError(double[][] dataset, Function f) {
        double err = 0.0;
        for (int i = 0; i < dataset.length; i++) {
            double x = dataset[i][0];
            double y = dataset[i][1];
            err += 0.5 * Math.pow((y - f.predict(x)), 2.0);
        }
        return Math.sqrt(2 * err / dataset.length);
    }

    private RealMatrix resolve(double[][] dataset, int m) {
        RealMatrix t = createColumnRealMatrix(dataset, 1);
        RealMatrix phi = MatrixUtils.createRealMatrix(dataset.length, m + 1);
        for (int i = 0; i < dataset.length; i++) {
            for (int j = 0; j < m + 1; j++) {
                phi.setEntry(i, j, Math.pow(dataset[i][0], j));
            }
        }
        RealMatrix tmp = MatrixUtils.inverse(phi.transpose().multiply(phi));
        RealMatrix ws = tmp.multiply(phi.transpose()).multiply(t);
        return ws;
    }

    private RealMatrix createColumnRealMatrix(double[][] dataset, int index) {
        double[] vals = new double[dataset.length];
        for (int i = 0; i < dataset.length; i++) {
            vals[i] = dataset[i][index];
        }
        return MatrixUtils.createColumnRealMatrix(vals);
    }

    public static void main(String[] args) {
        SquareError se = new SquareError(10, new int[]{0, 1, 3, 9});
        // 学習用と評価用のデータセットを作成
        double[][] trainSet = se.createDataset(se.N);
        double[][] testSet = se.createDataset(se.N);

        for (int m : se.M) {
            RealMatrix ws = se.resolve(trainSet, m);
            Function f = new Function() {
                @Override
                public double predict(double x) {
                    double ret = 0.0;
                    for (int i = 0; i < m + 1; i++) {
                        ret += ws.getEntry(i, 0) * Math.pow(x, i);
                    }
                    return ret;
                }
            };
            double trainError = se.rmsError(trainSet, f);
            double testError = se.rmsError(testSet, f);
            System.out.println("Train Error: " + trainError + ", Test Error: " + testError);
        }
    }
}

