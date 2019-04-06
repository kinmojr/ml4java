package ch04;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import static common.Compat.print;
import static common.Compat.dot;
import static common.Functions.*;

public class GradientSimpleNet {
    RealMatrix W;

    public GradientSimpleNet() {
        W = randn(2, 3);
    }

    public RealMatrix predict(RealMatrix x) {
        return dot(x, W);
    }

    public double loss(RealMatrix x, double[][] t) {
        RealMatrix z = predict(x);
        double[][] y = softmax(z);
        double loss = crossEntropyError(y, t);

        return loss;
    }

    public void main() {
        RealMatrix x = MatrixUtils.createRealMatrix(new double[][]{{0.6, 0.9}});
        double[][] t = new double[][]{{0, 0, 1}};

        Function f = new Function() {
            @Override
            public double function() {
                return loss(x, t);
            }
        };
        double[][] dW = numericalGradient(f, W);
        print(dW);
    }

    public static void main(String... args) {
        GradientSimpleNet simpleNet = new GradientSimpleNet();
        simpleNet.main();
    }

    double[][] numericalGradient(Function f, RealMatrix W) {
        double h = Math.pow(10, -4);
        double[][] grad = new double[W.getRowDimension()][W.getColumnDimension()];

        for (int i = 0; i < grad.length; i++) {
            for (int j = 0; j < grad[0].length; j++) {
                double tmpVal = W.getEntry(i, j);
                W.setEntry(i, j, tmpVal + h);
                double fxh1 = f.function();

                W.setEntry(i, j, tmpVal - h);
                double fxh2 = f.function();
                grad[i][j] = (fxh1 - fxh2) / (2 * h);

                W.setEntry(i, j, tmpVal);
            }
        }

        return grad;
    }

    interface Function {
        public double function();
    }
}
