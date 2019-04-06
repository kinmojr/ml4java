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

    private double[] resolve(double[][] dataset, int m) {
        RealMatrix t = createColumnRealMatrix(dataset, 1);
        double[] ws = new double[m];
        return ws;
    }

    private RealMatrix createColumnRealMatrix(double[][] dataset, int index) {
        double[] vals = new double[dataset.length];
        for (int i = 0; i < dataset.length; i++) {
            vals[i] = dataset[i][index];
        }
        RealMatrix ret = MatrixUtils.createColumnRealMatrix(vals);
        return ret;
    }

    public static void main(String[] args) {
        SquareError se = new SquareError(10, new int[]{0, 1, 3, 9});
        double[][] trainSet = se.createDataset(se.N);
        double[][] testSet = se.createDataset(se.N);

        for (int m : se.M) {

        }
//        int num = 100;
//        double[][] tmp = se.createDataset(num);
//        for (int i = 0; i < num; i++) {
//            System.out.print(tmp[i][0]);
//            System.out.print(", ");
//            System.out.println(tmp[i][1]);

//        }
    }
}
