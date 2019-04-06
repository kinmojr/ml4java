package ch03;

import static common.Compat.arrange;

public class Relu {
    public static double[] relu(double[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            if (x[i] < 0) ret[i] =  0;
            else ret[i] =  x[i];
        }
        return ret;
    }

    public static void main(String... args) {
        double[] x = arrange(-5.0, 5.0, 0.1);
        double[] y = relu(x);
        for (int i = 0; i < x.length; i++) {
            System.out.println(x[i] + "\t" + y[i]);
        }
    }
}
