package to.kishimo.ml4se.ch04;

public class Data {
    public double x;
    public double y;
    public int type;

    public Data(double[] mu, double[][] cov, int type) {
        this.type = type;
    }
}
