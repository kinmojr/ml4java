package to.kishimo.deeplearning;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class MatrixDotCommonsMath {
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
        RealMatrix matA = MatrixUtils.createRealMatrix(a);
        RealMatrix matB = MatrixUtils.createRealMatrix(b);
        long start = System.currentTimeMillis();
        ret = matA.multiply(matB).getData();
        System.out.println((System.currentTimeMillis() - start) / 1000.0);
        System.out.println(ret[999][999]);
    }
}
