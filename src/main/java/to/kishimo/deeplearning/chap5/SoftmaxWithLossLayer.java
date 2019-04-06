package to.kishimo.deeplearning.chap5;

import to.kishimo.deeplearning.chap3.Chap3Main;
import to.kishimo.deeplearning.chap4.Chap4Main;

public class SoftmaxWithLossLayer implements LastLayerFunction {
    double loss;
    double[][] y;
    int[][] t;

    @Override
    public double forward(double[][] x, int[][] t) {
        this.t = t;
        y = Chap3Main.softmax(x);
        loss = Chap4Main.crossEntropyErrorMean(y, t);
        return loss;
    }

    @Override
    public double[][] backward(double[][] dout) {
        double[][] dx = new double[y.length][y[0].length];
        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < y[i].length; j++) {
                dx[i][j] = (y[i][j] - t[i][j]) / y.length;
            }
        }
        return dx;
    }
}

