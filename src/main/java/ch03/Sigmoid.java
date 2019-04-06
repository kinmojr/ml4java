package ch03;

import static common.Compat.arrange;

public class Sigmoid {
    public static double[] sigmoid(double[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            ret[i] = 1.0 / (1.0 + java.lang.Math.exp(-x[i]));
        }
        return ret;
    }

    public static void main(String... args) {
        double[] x = arrange(-5.0, 5.0, 0.1);
        double[] y = sigmoid(x);
        for (int i = 0; i < x.length; i++) {
            System.out.println(x[i] + "\t" + y[i]);
        }
    }
}

