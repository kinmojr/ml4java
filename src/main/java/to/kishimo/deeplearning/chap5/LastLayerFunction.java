package to.kishimo.deeplearning.chap5;

public interface LastLayerFunction {
    public double forward(double[][] x, int[][] t);

    public double[][] backward(double dout[][]);
}
