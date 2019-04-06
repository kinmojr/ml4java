package to.kishimo.deeplearning.chap3;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;

/**
 * # 3.6.3 手書き数字認識 バッチ処理.
 * MNISTの手書き文字をニューラルネットワークで判別する.
 */
public class Chap3NeuralNetworkBatch {
    /**
     * バッチサイズ.
     */
    private static final int BATCH_SIZE = 100;

    /**
     * 重みデータ.
     */
    private RealMatrix[] w;

    /**
     * バイアスデータ.
     */
    private RealMatrix[] b;

    /**
     * 評価データ.
     */
    private RealMatrix features;

    /**
     * 正解ラベルデータ.
     */
    private int[] labels;

    /**
     * main関数.
     */
    public static void main(String... args) throws IOException, ClassNotFoundException {
        Chap3NeuralNetworkBatch neuralNetworkBatch = new Chap3NeuralNetworkBatch();
        neuralNetworkBatch.execute();
    }

    /**
     * データを読み込んで、インスタンスを初期化する.
     */
    public Chap3NeuralNetworkBatch() throws IOException, ClassNotFoundException {
        // 学習済みの重みデータを読み込む
        // 784*50
        RealMatrix w1 = Chap3Main.readWeights("mnist/w1.tsv");
        // 50*100
        RealMatrix w2 = Chap3Main.readWeights("mnist/w2.tsv");
        // 100*10
        RealMatrix w3 = Chap3Main.readWeights("mnist/w3.tsv");
        w = new RealMatrix[]{w1, w2, w3};

        // 学習済みのバイアスデータを読み込む
        // 1*50
        RealMatrix b1 = MatrixUtils.createRealMatrix(convert2d(new double[]{-0.0675032, 0.0695926, -0.0273047, 0.0225609, -0.220015, -0.220388, 0.0486264, 0.134992, 0.233426, -0.0487357, 0.101702, -0.0307604, 0.154824, 0.052125, 0.0601723, -0.0336486, -0.112183, -0.264607, -0.0332339, 0.136104, 0.0635437, 0.0467981, -0.0162165, -0.0577584, -0.0310868, 0.103662, -0.0845938, 0.116652, 0.218521, 0.0443726, 0.0337839, -0.0172038, -0.0738377, 0.161521, -0.106212, -0.0164695, 0.00913961, 0.102384, 0.00916639, -0.0564299, -0.106075, 0.0989272, -0.0713689, -0.0634913, 0.124617, 0.0224228, -0.000479718, 0.0452704, -0.151792, 0.107168,}, BATCH_SIZE));
        // 1*100
        RealMatrix b2 = MatrixUtils.createRealMatrix(convert2d(new double[]{-0.0147111, -0.0721513, -0.00155692, 0.121997, 0.116033, -0.00754946, 0.0408545, -0.0849616, 0.0289804, 0.0199724, 0.197708, 0.0436512, -0.0651873, -0.0522632, 0.0113163, 0.0304998, 0.0406035, 0.0695399, -0.0777847, 0.0692313, -0.0936553, 0.0548001, -0.0384374, 0.0212311, 0.0379341, -0.0280627, -0.0181841, 0.0687042, 0.0542943, 0.0674368, 0.0626431, -0.0233236, -0.0158914, 0.0186052, 0.0183929, -0.015681, -0.0742221, -0.0160673, -0.0226217, -0.0100751, 0.0434415, -0.120202, 0.0280247, -0.0759194, -0.00533499, -0.0893522, -0.0181419, 0.0330689, -0.0181271, -0.0768938, -0.0271541, -0.0384708, -0.0531547, -0.0215329, 0.0689824, 0.0243113, -0.00333816, 0.00817491, 0.039117, -0.0292462, 0.0718473, -0.00356748, 0.0224618, 0.0398798, -0.0492193, 0.0245428, 0.0587579, 0.0850544, -0.00190306, -0.0304428, -0.0638337, 0.0470311, -0.120055, 0.0357395, -0.0429339, 0.0328387, -0.0334773, -0.136591, -0.00123189, 0.000968316, 0.0459039, -0.025178, -0.0207398, 0.0200558, 0.010629, 0.0190294, -0.0104692, 0.0577789, 0.0473716, -0.0436276, 0.0745086, 0.0507795, 0.0664883, 0.04064, -0.00265163, 0.00576806, -0.0965246, -0.0513131, 0.0219969, -0.0435861,}, BATCH_SIZE));
        // 1*10
        RealMatrix b3 = MatrixUtils.createRealMatrix(convert2d(new double[]{-0.0602398, 0.00932628, -0.0135995, 0.0216713, 0.0107372, 0.066197, -0.0839734, -0.00912251, 0.00576962, 0.0532335,}, BATCH_SIZE));
        b = new RealMatrix[]{b1, b2, b3};

        // 評価データを読み込む
        Chap3MnistDataSet testDataSet = Chap3MnistDataSet.createInstance("test", Chap3MnistDataSet.TEST_IMAGE_FILE, Chap3MnistDataSet.TEST_LABEL_FILE);
        features = MatrixUtils.createRealMatrix(testDataSet.getFeatures());
        labels = testDataSet.getLabels();
    }

    /**
     * 学習済みの重みを使ってスコアを計算する.
     */
    private void execute() {
        // 推論して正解率を計算する
        int accuracyCount = 0;
        int labelIndex = 0;
        for (int i = 0; i < features.getRowDimension(); i += BATCH_SIZE) {
            RealMatrix batchFeature = features.getSubMatrix(i, i + BATCH_SIZE - 1, 0, features.getColumnDimension() - 1);
            double[][] predictions = predict(batchFeature, w, b);
            for (double[] prediction : predictions) {
                int maxIndex = getMaxIndex(prediction);
                if (labels[labelIndex] == maxIndex) {
                    accuracyCount++;
                }
                labelIndex++;
            }
        }
        System.out.println("Accuracy:" + (double) accuracyCount / (double) features.getRowDimension());
    }

    /**
     * スコアを計算する.
     *
     * @param x 特徴量
     * @param w 重み
     * @param b バイアス
     * @return スコア値の配列
     */
    private double[][] predict(RealMatrix x, RealMatrix[] w, RealMatrix[] b) {
        RealMatrix a1 = x.multiply(w[0]).add(b[0]);
        RealMatrix z1 = Chap3Main.sigmoid(a1);

        RealMatrix a2 = z1.multiply(w[1]).add(b[1]);
        RealMatrix z2 = Chap3Main.sigmoid(a2);

        RealMatrix a3 = z2.multiply(w[2]).add(b[2]);

        return Chap3Main.softmax(a3.getData());
    }

    /**
     * 最大スコアのインデックスを取得する.
     *
     * @param values スコアの配列
     * @return 最大スコアのインデックス
     */
    private int getMaxIndex(double[] values) {
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

    /**
     * 1次元配列を2次元配列に変換する.
     *
     * @param base      元になる1次元配列
     * @param batchSize 行数
     * @return 2次元配列
     */
    private double[][] convert2d(double[] base, int batchSize) {
        double[][] ret = new double[batchSize][base.length];
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(base, 0, ret[i], 0, base.length);
        }
        return ret;
    }
}
