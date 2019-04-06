package ch01;

import static common.Compat.*;

public class SinCosGraph {
    public static void main(String... args) {
        double[] x = arrange(0, 6, 0.1);
        double[] y1 = sin(x);
        double[] y2 = cos(x);
        for (int i = 0; i < x.length; i++) {
            System.out.println(x[i] + "\t" + y1[i] + "\t" + y2[i]);
        }
    }
}
