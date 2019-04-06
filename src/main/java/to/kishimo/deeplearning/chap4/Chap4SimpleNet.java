package to.kishimo.deeplearning.chap4;

import to.kishimo.deeplearning.chap3.Chap3Main;

public class Chap4SimpleNet {
    public double[][] w;

    public Chap4SimpleNet() {
        w = new double[][]{{0.47355232, 0.9977393, 0.84668094}, {0.85557411, 0.03563661, 0.69422093}};
    }

    public double[][] predict(double[][] x) {
        return Chap3Main.dotProduct(x, w);
    }

    public double loss(double[][] x, double[] t) {
        double[][] z = predict(x);
        double[][] y = Chap3Main.softmax(z);
        double loss = Chap4Main.crossEntropyError(y[0], t);
        return loss;
    }

    public int argMax(double[] values) {
        int index = 0;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > max) {
                max = values[i];
                index = i;
            }
        }
        return index;
    }
}

