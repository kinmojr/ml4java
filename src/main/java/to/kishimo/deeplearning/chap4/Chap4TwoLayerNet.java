package to.kishimo.deeplearning.chap4;

import to.kishimo.deeplearning.chap3.Chap3Main;
import to.kishimo.deeplearning.chap3.Chap3MnistDataSet;

import java.util.Random;
import java.util.concurrent.*;

public class Chap4TwoLayerNet {
    private int inputSize, hiddenSize, outputSize;
    private double weightInitStd, learnRate;
    private double[][] paramW1, paramW2, gradW1, gradW2;
    private double[] paramB1, paramB2, gradB1, gradB2;

    public Chap4TwoLayerNet(int inputSize, int hiddenSize, int outputSize, double weightInitStd, double learnRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.weightInitStd = weightInitStd;
        this.learnRate = learnRate;

        Random rand = new Random();
        paramW1 = new double[inputSize][hiddenSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                paramW1[i][j] = weightInitStd * rand.nextGaussian();
            }
        }

        paramB1 = new double[hiddenSize];

        paramW2 = new double[hiddenSize][outputSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                paramW2[i][j] = weightInitStd * rand.nextGaussian();
            }
        }

        paramB2 = new double[outputSize];
    }

    /**
     * 予測.
     *
     * @param x
     * @return
     */
    private double[][] predict(double[][] x) {
        double[][] a1 = Chap3Main.addBias(Chap3Main.dotProduct(x, paramW1), paramB1);
        double[][] z1 = Chap3Main.sigmoid(a1);
        double[][] a2 = Chap3Main.addBias(Chap3Main.dotProduct(z1, paramW2), paramB2);
        double[][] y = Chap3Main.softmax(a2);
        return y;
    }

    /**
     * 平均クロスエントロピー誤差.
     *
     * @param y
     * @param t
     * @return
     */
    private double crossEntropyErrorMean(double[][] y, int[] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += -Math.log(y[i][t[i]] + Math.pow(10.0, -7.0));
        }
        return sum / y.length;
    }

    /**
     * 損失関数.
     *
     * @param x
     * @param t
     * @return
     */
    private double loss(double[][] x, int[] t) {
        double[][] y = predict(x);
        double loss = crossEntropyErrorMean(y, t);
        return loss;
    }

    /**
     * 重み行列に対する数値勾配を求める.
     *
     * @param w
     * @param x
     * @param t
     * @return
     * @throws InterruptedException
     * @throws TimeoutException
     * @throws ExecutionException
     */
    private double[][] numericalGradientMatrix(double[][] w, double[][] x, int[] t) throws InterruptedException, TimeoutException, ExecutionException {
//        double[][] ret = new double[w.length][w[0].length];
//        for (int i = 0; i < ret.length; i++) {
//            ret[i] = numericalGradientVector(w[i], x, t);
//        }
//        return ret;

        double[][] ret = new double[w.length][w[0].length];
        ExecutorService executor = Executors.newFixedThreadPool(8);
        Future<double[]>[] futures = new Future[ret.length];
        for (int i = 0; i < ret.length; i++) {
            double[] vector = w[i];
            futures[i] = executor.submit(new Callable<double[]>() {
                @Override
                public double[] call() throws Exception {
                    return numericalGradientVector(vector, x, t);
                }
            });
        }
        for (int i = 0; i < ret.length; i++) {
            ret[i] = futures[i].get(300, TimeUnit.SECONDS);
        }
        executor.shutdown();

        return ret;
    }

    /**
     * 重みベクトルに対する数値購買を求める.
     *
     * @param w
     * @param x
     * @param t
     * @return
     * @throws InterruptedException
     * @throws TimeoutException
     * @throws ExecutionException
     */
    private double[] numericalGradientVector(double[] w, double[][] x, int[] t) throws InterruptedException, TimeoutException, ExecutionException {
        double[] ret = new double[w.length];
        for (int i = 0; i < ret.length; i++) {
            double diff = Math.pow(10, -4);
            double tmp = w[i];

            w[i] = tmp + diff;
            double value1 = loss(x, t);

            w[i] = tmp - diff;
            double value2 = loss(x, t);

            ret[i] = (value1 - value2) / (2 * diff);
            w[i] = tmp;
        }
        return ret;
    }

    /**
     * 学習して重みを更新.
     *
     * @param x
     * @param t
     * @throws InterruptedException
     * @throws TimeoutException
     * @throws ExecutionException
     */
    private void learnUpdate(double[][] x, int[] t) throws InterruptedException, TimeoutException, ExecutionException {
        gradW1 = numericalGradientMatrix(paramW1, x, t);
        gradW2 = numericalGradientMatrix(paramW2, x, t);
        gradB1 = numericalGradientVector(paramB1, x, t);
        gradB2 = numericalGradientVector(paramB2, x, t);

        updateWeights(paramW1, gradW1);
        updateBias(paramB1, gradB1);
        updateWeights(paramW2, gradW2);
        updateBias(paramB2, gradB2);
    }

    private void updateWeights(double[][] paramW, double[][] gradW) {
        for (int i = 0; i < paramW.length; i++) {
            for (int j = 0; j < paramW[0].length; j++) {
                if (gradW[i][j] != 0.0) {
                    paramW[i][j] += -learnRate * gradW[i][j];
                }
            }
        }
    }

    private void updateBias(double[] paramB, double[] gradB) {
        for (int i = 0; i < paramB.length; i++) {
            if (gradB[i] != 0.0) {
                paramB[i] += -learnRate * gradB[i];
            }
        }
    }

    private double accuracy(Chap3MnistDataSet dataSet) {
        double[][] features = dataSet.getFeatures();
        int[] labels = dataSet.getLabels();
        double[][] y = predict(features);
        int count = 0;
        for (int i = 0; i < labels.length; i++) {
            if (getMaxIndex(y[i]) == labels[i]) {
                count++;
            }
        }
        return (double) count / (double) features.length;
    }

    private static int getMaxIndex(double[] values) {
        double maxValue = Double.NaN;
        int maxIndex = 0;
        for (int i = 0; i < values.length; i++) {
            if (Double.isNaN(maxValue)) {
                maxValue = values[i];
            } else if (values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static void main(String... args) throws Exception {
        Chap3MnistDataSet trainDataSet = Chap3MnistDataSet.createInstance("train");
        Chap3MnistDataSet testDataSet = Chap3MnistDataSet.createInstance("test");
        int itersNum = 10000;
        int batchSize = 100;

        Chap4TwoLayerNet network = new Chap4TwoLayerNet(784, 50, 10, 0.01, 0.1);
        System.out.println("Index\tTime(sec)\tAccuracy(train)\tAccuracy(test)\tLoss(train)\tLoss(test)");
        for (int i = 0; i < itersNum; i++) {
            long start = System.currentTimeMillis();
            int[] indexes = trainDataSet.getRandomIndexes(batchSize);
            double[][] batchX = trainDataSet.getRandomFeatures(indexes);
            int[] batchT = trainDataSet.getRandomLabels(indexes);
            network.learnUpdate(batchX, batchT);
            System.out.print(i + "\t" + (System.currentTimeMillis() - start) / 1000 + "\t");
            System.out.print(network.accuracy(trainDataSet) + "\t");
            System.out.print(network.accuracy(testDataSet) + "\t");
            System.out.print(network.loss(testDataSet.getFeatures(), testDataSet.getLabels()) + "\t");
            System.out.println(network.loss(batchX, batchT));
        }
    }
}
