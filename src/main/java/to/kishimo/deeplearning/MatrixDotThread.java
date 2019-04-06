package to.kishimo.deeplearning;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MatrixDotThread implements Runnable {
    private static int SIZE = 1000;

    private static double[][] ret = new double[SIZE][SIZE];
    private static double[][] a = new double[SIZE][SIZE];
    private static double[][] b = new double[SIZE][SIZE];

    private int index = 0;

    static {
        Random rand = new Random(1);
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                a[i][j] = rand.nextDouble();
                b[i][j] = rand.nextDouble();
            }
        }
    }

    public MatrixDotThread(int index) {
        this.index = index;
    }

    public void run() {
        dotProduct(index);
    }

    public static void main(String... args) throws InterruptedException {
        System.out.println("start");
        long start = System.currentTimeMillis();

        ExecutorService executer = Executors.newFixedThreadPool(20);

        for (int i = 0; i < SIZE; i++) {
            executer.execute(new MatrixDotThread(i));
        }

        executer.shutdown();
        boolean flag = executer.awaitTermination(100, TimeUnit.SECONDS);

        System.out.println((System.currentTimeMillis() - start) / 1000.0);
        if (flag) {
            System.out.println("finish comp");
        } else {
            System.out.println("finish imcomp");
        }
        System.out.println(ret[999][999]);
    }

    public static void dotProduct(int index) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                ret[index][j] += a[index][k] * b[k][j];
            }
        }
    }
}

