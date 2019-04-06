package ch04;

import static common.Compat.*;

public class Gradient2D {
    static double[] _numericalGradientNoBatch(Function2 f, double[] x) {
        double h = Math.pow(10, -4);
        double[] grad = new double[x.length];

        for (int idx = 0; idx < x.length; idx++) {
            double tmpVal = x[idx];
            x[idx] = tmpVal + h;
            double fxh1 = f.function(x);

            x[idx] = tmpVal - h;
            double fxh2 = f.function(x);
            grad[idx] = (fxh1 - fxh2) / (2 * h);

            x[idx] = tmpVal;
        }

        return grad;
    }

    static double[][] numericalGradient(Function2 f, double[][] X) {
        double[][] grad = new double[X.length][X[0].length];

        for (int idx = 0; idx < X.length; idx++) {
            grad[idx] = _numericalGradientNoBatch(f, X[idx]);
        }

        return grad;
    }

    private static double function2(double[] x) {
        double ret = 0.0;
        for (int i = 0; i < x.length; i++) {
            ret += Math.pow(x[i], 2.0);
        }
        return ret;
    }

    public static void main(String... args) {
        double[] x0 = arrange(-2, 2.5, 0.25);
        double[] x1 = arrange(-2, 2.5, 0.25);
        double[][][] XY = meshgrid(x0, x1);

        double[] X = flatten(XY[0]);
        double[] Y = flatten(XY[1]);

        Function2 function2 = new Function2() {
            @Override
            public double function(double[] x) {
                return function2(x);
            }
        };

        double[][] grad = numericalGradient(function2, new double[][]{X, Y});

        for (int j = 0; j < grad[0].length; j++) {
            System.out.println("x" + j + " = " + -grad[0][j] + ", y" + j + " = " + -grad[1][j]);
        }
    }

    interface Function2 {
        public double function(double[] x);
    }
}

