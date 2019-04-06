package ch01;

import static common.Compat.arrange;
import static common.Compat.sin;

public class SimpleGraph {
    public static void main(String... args) {
        double[] x = arrange(0, 6, 0.1);
        double[] y = sin(x);
        for (int i = 0; i < x.length; i++) {
            System.out.println(x[i] + "\t" + y[i]);
        }
    }
}
