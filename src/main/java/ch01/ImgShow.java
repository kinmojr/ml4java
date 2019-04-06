package ch01;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

import java.io.File;
import java.net.MalformedURLException;

public class ImgShow extends Application {
    @Override
    public void start(Stage stage) throws MalformedURLException {
        ImageView imageView = new ImageView(new Image(new File("dataset/lena.png").toURI().toURL().toString()));

        StackPane root = new StackPane();
        root.getChildren().add(imageView);

        Scene scene = new Scene(root, 256, 256);
        stage.setScene(scene);

        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
