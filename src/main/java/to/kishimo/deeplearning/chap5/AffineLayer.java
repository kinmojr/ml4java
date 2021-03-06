package to.kishimo.deeplearning.chap5;

import to.kishimo.deeplearning.chap3.Chap3Main;

public class AffineLayer implements HiddenLayerFunction {
    double[][] W;
    double[] b;
    double[][] x;
    double[][] dW;
    double[] db;

    public AffineLayer(double[][] W, double[] b) {
        this.W = W;
        this.b = b;
    }

    @Override
    public double[][] forward(double[][] x) {
        this.x = x;
        double[][] out = Chap3Main.addBias(Chap3Main.dotProduct(x, W), b);
        return out;
    }

    @Override
    public double[][] backward(double[][] dout) {
        double[][] dx = Chap3Main.dotProduct(dout, transpose(W));
        dW = Chap3Main.dotProduct(transpose(x), dout);
        db = sum(dout);
        return dx;
    }

    @Override
    public double[][] getGradWeights(){
        return dW;
    }

    @Override
    public double[] getGradBias(){
        return db;
    }

    private double[][] transpose(double[][] matrix) {
        double[][] ret = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                ret[j][i] = matrix[i][j];
            }
        }
        return ret;
    }

    private double[] sum(double[][] matrix) {
        double[] ret = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < matrix[0].length; j++) {
                sum += matrix[i][j];
            }
            ret[i] = sum;
        }
        return ret;
    }
}

