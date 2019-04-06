package to.kishimo.deeplearning.common.layers;

public interface Layer {
    public double[][] forward(double[][] x);
    
    public double[][] backward(double[][] dout);

    public double[][] dW();

    public double[][] db();

    public double[][] W();

    public double[][] b();
}
