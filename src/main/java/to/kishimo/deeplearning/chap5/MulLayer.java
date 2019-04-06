package to.kishimo.deeplearning.chap5;

public class MulLayer {
    double x;
    double y;

    public MulLayer(double x, double y) {
        this.x = x;
        this.y = y;
    }

    double forward(double x, double y) {
        this.x = x;
        this.y = y;
        double out = x * y;
        return out;
    }

    double[] backward(double dout) {
        double dx = dout * this.y;
        double dy = dout * this.x;
        return new double[]{dx, dy};
    }
}

