package ch04;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Test {
    public static void main(String... args) {
        RealMatrix a = MatrixUtils.createRealMatrix(new double[][]{{1, 2}, {3, 4}});
        RealMatrix b = MatrixUtils.createRealMatrix(new double[][]{{5, 6}, {7, 8}});
        RealMatrix c = a.multiply(b);
        double[][] c2 = c.getData();
    }
}
