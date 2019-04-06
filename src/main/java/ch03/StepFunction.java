package ch03;

import static common.Compat.arrange;
import static common.Compat.cos;
import static common.Compat.sin;

public class StepFunction {
    public static double[] stepFunction(double[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            if (x[i] > 0.0) ret[i] = 1.0;
            else ret[i] = 0.0;
        }
        return ret;
    }

    public static void main(String... args) {
        double[] x = arrange(-5.0, 5.0, 0.1);
        double[] y = stepFunction(x);
        for (int i = 0; i < x.length; i++) {
            System.out.println(x[i] + "\t" + y[i]);
        }
    }
}
