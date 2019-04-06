package to.kishimo.deeplearning;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MatrixDot {
    private static int SIZE = 1000;

    private static double[][] ret = new double[SIZE][SIZE];
    private static double[][] a = new double[SIZE][SIZE];
    private static double[][] b = new double[SIZE][SIZE];

    static {
        Random rand = new Random(1);
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                a[i][j] = rand.nextDouble();
                b[i][j] = rand.nextDouble();
            }
        }
    }

    public static void main(String... args) throws InterruptedException {
        long start = System.currentTimeMillis();
        ret = dotProduct(a, b);
        System.out.println((System.currentTimeMillis() - start) / 1000.0);
        System.out.println(ret[999][999]);
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
}
