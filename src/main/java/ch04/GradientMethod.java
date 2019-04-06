package ch04;

import ch04.Gradient2D.Function2;

import static ch04.Gradient2D._numericalGradientNoBatch;

public class GradientMethod {
    private static double[] gradientDescent(Function2 f, double[] initX, double lr, int stepNum) {
        double[] x = initX;
        for (int i = 0; i < stepNum; i++) {
            double[] grad = _numericalGradientNoBatch(f, x);
            for (int j = 0; j < x.length; j++) {
                x[j] -= lr * grad[j];
            }
        }
        return x;
    }

    private static double function2(double[] x) {
        return Math.pow(x[0], 2.0) + Math.pow(x[1], 2.0);
    }

    public static void main(String... args) {
        double[] initX = new double[]{-3.0, 4.0};

        double lr = 0.1;
        int stepNum = 20;
        Function2 function2 = new Function2() {
            @Override
            public double function(double[] x) {
                return function2(x);
            }
        };
        double[] x = gradientDescent(function2, initX, lr, stepNum);

        System.out.println("x1 = " + x[0] + ", x2 = " + x[1]);
    }
}

