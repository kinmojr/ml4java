package to.kishimo.ml4se.ch05;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.*;

public class RocCurve {
    private static double[] variances = {50.0, 150.0};

    public static void main(String[] args) {
        for (double variance : variances) {
            System.out.println("Variance: " + variance);
            RocCurve rc = new RocCurve();
            List<Point> result = rc.runSimulation(variance);
            rc.calcRocAuc(result);
            System.out.println();
        }
    }

    private List<Point> runSimulation(double variance) {
        List<Point> trainSet = prepareDataset(variance);
        RealMatrix w = MatrixUtils.createRealMatrix(new double[][]{{0.0}, {0.1}, {0.1}});
        RealMatrix phi = createPhi(trainSet);
        RealMatrix t = createT(trainSet);
        for (int i = 0; i < 100; i++) {
            RealMatrix y = MatrixUtils.createRealMatrix(1, phi.getRowDimension());
            for (int j = 0; j < phi.getRowDimension(); j++) {
                RealMatrix line = phi.getRowMatrix(j);
                RealMatrix a = line.multiply(w);
                y.setEntry(0, j, 1.0 / (1.0 + Math.exp(-a.getEntry(0, 0))));
            }
            RealMatrix r = MatrixUtils.createRealDiagonalMatrix(y.getRow(0));
            y = y.transpose();
            RealMatrix tmp1 = MatrixUtils.inverse(phi.transpose().multiply(r).multiply(phi));
            RealMatrix tmp2 = phi.transpose().multiply(y.subtract(t));
            RealMatrix wNew = w.subtract(tmp1.multiply(tmp2));
            if (wNew.subtract(w).transpose().multiply(wNew.subtract(w)).getEntry(0, 0) < 0.001 * w.transpose().multiply(w).getEntry(0, 0)) {
                w = wNew;
                break;
            }
            w = wNew;
        }
        // 不正解率を計算する
        double err = 0.0;
        for (Point line : trainSet) {
            double a = w.getEntry(0, 0) + w.getEntry(1, 0) * line.x + w.getEntry(2, 0) * line.y;
            double p = 1.0 / (1.0 + Math.exp(-a));
            line.prob = p;
            if ((p - 0.5) * (line.type * 2 - 1) < 0) {
                err += 1.0;
            }
        }
        double errRate = err * 100 / trainSet.size();
        System.out.println("Error Rate: " + errRate + "%");
        return trainSet;
    }

    private void calcRocAuc(List<Point> result) {
        // 確率値が高い順にソートする
        result.sort(
                new Comparator<Point>() {
                    @Override
                    public int compare(Point obj1, Point obj2) {
                        if (obj2.prob - obj1.prob > 0.0) {
                            return 1;
                        } else if (obj2.prob - obj1.prob == 0.0) {
                            return 0;
                        } else {
                            return -1;
                        }
                    }
                }
        );
        // 陽性数と陰性数をカウントする
        int positives = 0;
        int negatives = 0;
        for (int i = 0; i < result.size(); i++) {
            Point line = result.get(i);
            if (line.type == 1) positives++;
            else negatives++;
        }
        // 真陽性率と偽陽性率を計算する
        double[] tprs = new double[result.size()];
        double[] fprs = new double[result.size()];
        for (int i = 0; i < result.size(); i++) {
            Point line = result.get(i);
            for (int j = 0; j < result.size(); j++) {
                if (i < j) {
                    if (line.type == 1) {
                        tprs[j] += 1.0 / positives;
                    } else {
                        fprs[j] += 1.0 / negatives;
                    }
                }
            }
        }
        // ROC AUCを計算する
        double rocAuc = 0.0;
        double lastFpr = 0.0;
        for (int i = 0; i < tprs.length; i++) {
            if (fprs[i] > lastFpr || i == (tprs.length - 1)) {
                rocAuc += tprs[i] * (fprs[i] - lastFpr);
                lastFpr = fprs[i];
            }
        }
        System.out.println("ROC AUC: " + rocAuc);
    }

    /**
     * 学習・評価データを作成する.
     * @param var 学習・評価データの分散
     * @return データのリスト
     */
    private List<Point> prepareDataset(double var) {
        int n1 = 80;
        int n2 = 200;
        double[] mu1 = {9.0, 9.0};
        double[] mu2 = {-3.0, -3.0};
        List<Point> df1 = variateNormal(mu1, var, 1, n1);
        List<Point> df2 = variateNormal(mu2, var, 0, n2);
        List<Point> df = new ArrayList<>();
        df.addAll(df1);
        df.addAll(df2);
        Collections.shuffle(df);
        return df;
    }

    /**
     * φ行列を作成する.
     * @param trainSet 学習・評価データセット
     * @return φ行列
     */
    private RealMatrix createPhi(List<Point> trainSet) {
        RealMatrix phi = MatrixUtils.createRealMatrix(trainSet.size(), 3);
        for (int i = 0; i < trainSet.size(); i++) {
            phi.setRow(i, new double[]{1.0, trainSet.get(i).x, trainSet.get(i).y});
        }
        return phi;
    }

    /**
     * 陽性・陰性ラベルの行列を作成する.
     * @param trainSet 学習・評価データセット
     * @return 陽性・陰性ラベルの行列
     */
    private RealMatrix createT(List<Point> trainSet) {
        RealMatrix t = MatrixUtils.createRealMatrix(trainSet.size(), 1);
        for (int i = 0; i < trainSet.size(); i++) {
            t.setEntry(i, 0, trainSet.get(i).type);
        }
        return t;
    }

    /**
     * ばらつきのあるデータセットを作成する.
     * @param mu 平均(中心)
     * @param var 分散
     * @param type 陽性・陰性ラベル
     * @param num 数
     * @return データセット
     */
    private List<Point> variateNormal(double[] mu, double var, int type, int num) {
        List<Point> dataset = new ArrayList<>();
        for (int i = 0; i < num; i++) {
            Point p = new Point(mu, var, type);
            dataset.add(p);
        }
        return dataset;
    }

    /**
     * x、y座標、陽性・陰性ラベルを持つデータ.
     */
    private static class Point {
        private static Random rand = new Random();

        private double x;
        private double y;
        private int type;
        private double prob;

        private Point(double[] mu, double var, int type) {
            this.type = type;
            x = mu[0] + rand.nextGaussian() * Math.sqrt(var);
            y = mu[1] + rand.nextGaussian() * Math.sqrt(var);
            prob = 0.0;
        }
    }
}
