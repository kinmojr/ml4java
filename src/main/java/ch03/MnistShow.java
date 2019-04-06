package ch03;

import dataset.Mnist;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.HashMap;

public class MnistShow {
    public static void main(String... args) throws IOException, ClassNotFoundException {
        HashMap<String, double[][]> dataset = new Mnist().loadMinist(false, true, false);
        double[] image = dataset.get("train_img")[0];
        int label = (int) dataset.get("train_label")[0][0];
        System.out.println(label);

        new MnistShow().imgShow(image, label);
    }

    public void imgShow(double[] image, int label) {
        BufferedImage bufImg = makeImage(image);
        Icon icon = new ImageIcon(bufImg);
        try {
            JOptionPane.showMessageDialog(null, label, "MnistImageViewer", JOptionPane.PLAIN_MESSAGE, icon);
        } catch (HeadlessException e) {
            System.out.println("Image dialog can't be displayed on CUI environment.");
        }
    }

    private BufferedImage makeImage(double[] image) {
        BufferedImage bufImg =
                new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int value = (int) (image[i * 28 + j]);
                bufImg.setRGB(j, i, 0xff000000 | value << 16 | value << 8 | value);
            }
        }

        return bufImg;
    }
}

