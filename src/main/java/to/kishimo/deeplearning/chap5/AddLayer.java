package to.kishimo.deeplearning.chap5;

public class AddLayer {
    double forward(double x, double y) {
        double out = x + y;
        return out;
    }

    double[] backward(double dout) {
        double dx = dout * 1;
        double dy = dout * 1;
        return new double[]{dx, dy};
    }
}

