package to.kishimo.deeplearning.common.layers;

public interface LastLayer {
    public double forward(double[][] x, int[][] t);

    public double[][] backward(double dout[][]);
}

