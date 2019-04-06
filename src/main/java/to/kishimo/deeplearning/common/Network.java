package to.kishimo.deeplearning.common;

import to.kishimo.deeplearning.common.layers.*;

import java.util.*;

public interface Network {
    public double[][] predict(double[][] x);

    public double loss(double[][] x, int[][] t);

    public double accuracy(double[][] x, int[][] t);

    public LinkedHashMap<String, double[][]> gradient(double[][] x, int[][] t);

    public Map<String, double[][]> params();
}
