package to.kishimo.ml4se.ch04;

import java.util.Random;

public class Point {
    private static Random rand = new Random();

    public double x;
    public double y;
    public int type;

    public Point(double[] mu, double var, int type) {
        this.type = type;
        x = mu[0] + rand.nextGaussian() * Math.sqrt(var);
        y = mu[1] + rand.nextGaussian() * Math.sqrt(var);
    }
}
