package to.kishimo.ml4se.chap02;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class SquareError {
    private static Random rand = new Random();

    public static void main(String[] args) {
        int N = 10;
        int[] M = new int[]{0, 1, 3, 9};

        SquareError se = new SquareError();
        RealMatrix trainSet = se.createDataset(N);
        RealMatrix testSet = se.createDataset(N);

        for (int m : M) {
            RealMatrix ws = se.resolve(trainSet, m);
            double trainError = se.rmsError(trainSet, ws);
            double testError = se.rmsError(testSet, ws);
            System.out.println("Train Error: " + trainError + ", Test Error: " + testError);
        }
        System.out.println();

        for (int m = 0; m < N; m++) {
            RealMatrix ws = se.resolve(trainSet, m);
            double trainError = se.rmsError(trainSet, ws);
            double testError = se.rmsError(testSet, ws);
            System.out.println("Train Error: " + trainError + ", Test Error: " + testError);
        }
    }

    private RealMatrix createDataset(int rows) {
        RealMatrix ret = MatrixUtils.createRealMatrix(rows, 2);
        for (int i = 0; i < rows; i++) {
            double x = (double) i / (double) (rows - 1);
            double y = Math.sin(2 * Math.PI * x) + rand.nextGaussian() * 0.3;
            ret.setEntry(i, 0, x);
            ret.setEntry(i, 1, y);
        }
        return ret;
    }

    private double rmsError(RealMatrix dataset, RealMatrix ws) {
        double err = 0.0;
        for (int i = 0; i < dataset.getRowDimension(); i++) {
            double x = dataset.getEntry(i, 0);
            double y = dataset.getEntry(i, 1);
            err += 0.5 * Math.pow((y - predict(x, ws)), 2.0);
        }
        return Math.sqrt(2 * err / dataset.getRowDimension());
    }

    private double predict(double x, RealMatrix ws) {
        double ret = 0.0;
        for (int i = 0; i < ws.getRowDimension(); i++) {
            double w = ws.getEntry(i, 0);
            ret += w * Math.pow(x, i);
        }
        return ret;
    }

    private RealMatrix resolve(RealMatrix dataset, int m) {
        RealMatrix t = dataset.getColumnMatrix(1);
        RealMatrix phi = MatrixUtils.createRealMatrix(dataset.getRowDimension(), m + 1);
        for (int i = 0; i < dataset.getRowDimension(); i++) {
            for (int j = 0; j < m + 1; j++) {
                phi.setEntry(i, j, Math.pow(dataset.getEntry(i, 0), j));
            }
        }
        RealMatrix tmp = MatrixUtils.inverse(phi.transpose().multiply(phi));
        return tmp.multiply(phi.transpose()).multiply(t);
    }
}
