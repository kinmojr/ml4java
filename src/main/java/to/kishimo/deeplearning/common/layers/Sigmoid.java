package to.kishimo.deeplearning.common.layers;

import to.kishimo.deeplearning.common.layers.Layer;

public class Sigmoid implements Layer {
    double[][] out;

    @Override
    public double[][] forward(double[][] x) {
        double[][] out = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                out[i][j] = 1 / (1 + Math.exp(-x[i][j]));
            }
        }
        this.out = out;
        return out;
    }

    @Override
    public double[][] backward(double[][] dout) {
        double[][] dx = new double[dout.length][dout[0].length];
        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dx[i][j] = dout[i][j] * (1.0 - out[i][j]) * out[i][j];
            }
        }
        return dx;
    }

    @Override
    public double[][] dW() {
        return null;
    }

    @Override
    public double[][] W() {
        return null;
    }

    @Override
    public double[][] db() {
        return null;
    }

    @Override
    public double[][] b() {
        return null;
    }
}
