package to.kishimo.deeplearning;

public class MyThread implements Runnable {
    private final int id;

    MyThread(int id) {
        this.id = id;
    }

    public void run() {
        for (int i = 0; i < 20; i++) {
            System.out.println(id + " " + i);
        }
    }

    public static void main(String... args) throws InterruptedException {
        System.out.println("start");

        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(new MyThread(i));
            threads[i].start();
        }

        for (int i = 0; i < 20; i++) {
            System.out.println("10 " + i);
        }

        for (Thread t : threads) {
            t.join();
        }

        System.out.println("finish");
    }
}

