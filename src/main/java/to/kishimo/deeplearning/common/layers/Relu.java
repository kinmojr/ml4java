package to.kishimo.deeplearning.common.layers;

import to.kishimo.deeplearning.common.layers.Layer;

public class Relu implements Layer {
    boolean[][] mask;

    @Override
    public double[][] forward(double[][] x) {
        mask = new boolean[x.length][x[0].length];
        double[][] ret = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                if (x[i][j] <= 0) {
                    mask[i][j] = true;
                    ret[i][j] = 0.0;
                } else {
                    ret[i][j] = x[i][j];
                }
            }
        }
        return ret;
    }

    @Override
    public double[][] backward(double[][] dout) {
        double[][] ret = new double[dout.length][dout[0].length];
        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                if (mask[i][j]) {
                    ret[i][j] = 0;
                } else {
                    ret[i][j] = dout[i][j];
                }
            }
        }
        return ret;
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
