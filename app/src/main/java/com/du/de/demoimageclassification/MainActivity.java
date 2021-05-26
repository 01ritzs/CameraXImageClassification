package com.du.de.demoimageclassification;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;

import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    protected TextView tvMessage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_x_camera);
        tvMessage = findViewById(R.id.tvMessage);
        ImageClassifier.ImageClassifierOptions options = ImageClassifier.ImageClassifierOptions.builder().setMaxResults(1)
                .build();
        ImageClassifier imageClassifier = null;
        try {
            imageClassifier = ImageClassifier.createFromFileAndOptions(this, "FlowerModel.tflite", options);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.imag);
        TensorImage inputImage = TensorImage.fromBitmap(bitmap);


        List<Classifications> results = imageClassifier.classify(inputImage);
        if (results != null && results.get(0) != null) {
            List<Category> categories = results.get(0).getCategories();
            if (categories != null && categories.get(0) != null) {
                String msg = categories.get(0).getLabel() + " " + categories.get(0).getScore();
                Log.i("Check 1", msg);
                tvMessage.setText(msg);
            }
        }
    }
}