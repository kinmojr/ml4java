package to.kishimo.deeplearning.chap5;

public interface HiddenLayerFunction {
    public double[][] forward(double[][] x);
    
    public double[][] backward(double[][] dout);

    public double[][] getGradWeights();

    public double[] getGradBias();
}
