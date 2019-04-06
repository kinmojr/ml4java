package to.kishimo.deeplearning;

import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MyThreadPool implements Runnable {
    private final int id;

    MyThreadPool(int id) {
        this.id = id;
    }

    public void run() {
        for (int i = 0; i < 100; i++) {
            System.out.println(id + " " + i);
        }
    }

    public static void main(String... args) throws InterruptedException {
        System.out.println("start");

        ExecutorService executer = Executors.newFixedThreadPool(10);

        for (int i = 0; i < 100; i++) {
            executer.execute(new MyThreadPool(i));
        }

        for (int i = 0; i < 100; i++) {
            System.out.println("10 " + i);
        }

        executer.shutdown();
        executer.awaitTermination(1, TimeUnit.NANOSECONDS);

        System.out.println("finish");
    }
}

