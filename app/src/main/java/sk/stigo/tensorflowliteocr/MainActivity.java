package sk.stigo.tensorflowliteocr;

import static sk.stigo.tensorflowliteocr.utils.PhotoUtils.deletePhoto;
import static sk.stigo.tensorflowliteocr.utils.PhotoUtils.getRotatedImg;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.google.android.material.dialog.MaterialAlertDialogBuilder;

import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import sk.stigo.tensorflowliteocr.utils.OCRModelExecutor;

public class MainActivity extends AppCompatActivity {

    private ImageView photoResultImageView;
    private TextView recognitionResult;
    private TextView recognitionResultLabel;
    private String currentPhotoPath;
    private Button btnRecognize;
    private Button btnTakePicture;
    private ProgressBar spinner;
    private OCRModelExecutor ocrModelExecutor;
    private SharedPreferences.OnSharedPreferenceChangeListener sharedPreferenceChangeListener;
    private String selectedImageFile;


    static final int REQUEST_IMAGE_CAPTURE = 1;

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_items, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_settings:
                startActivity(new Intent(getApplicationContext(), SettingsActivity.class));
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Toolbar toolbar = findViewById(R.id.toolbar);
        photoResultImageView = findViewById(R.id.photoResult);
        recognitionResult = findViewById(R.id.recognitionResult);
        recognitionResultLabel = findViewById(R.id.recognitionResultLabel);
        btnRecognize = findViewById(R.id.btnRecognize);
        btnTakePicture = findViewById(R.id.btnTakePicture);
        spinner = findViewById(R.id.loadingSpinner);

        setSupportActionBar(toolbar);

        OpenCVLoader.initDebug();

        try {
            ocrModelExecutor = new OCRModelExecutor(getApplicationContext());
        } catch (IOException e) {
            e.printStackTrace();
            createErrorDialog("Loading TF model failed");
        }

        sharedPreferenceChangeListener = (sharedPreferences, s) -> {
            resetActivityViews(sharedPreferences);
        };

        SharedPreferences preferences = PreferenceManager.getDefaultSharedPreferences(getApplicationContext()) ;
        preferences.registerOnSharedPreferenceChangeListener(sharedPreferenceChangeListener);

        resetActivityViews(preferences);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        deletePhoto(currentPhotoPath);
        SharedPreferences preferences = PreferenceManager.getDefaultSharedPreferences(getApplicationContext()) ;
        preferences.unregisterOnSharedPreferenceChangeListener(sharedPreferenceChangeListener);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE) {
            if (resultCode == RESULT_OK) {
                setPic();
                if (btnRecognize.getVisibility() != View.VISIBLE) {
                    btnRecognize.setVisibility(View.VISIBLE);
                }
                if (recognitionResult.getVisibility() == View.VISIBLE) {
                    if (recognitionResult.getText() != null) {
                        recognitionResult.setText(null);
                    }
                    recognitionResult.setVisibility(View.GONE);
                    recognitionResultLabel.setVisibility(View.GONE);
                }
            } else {
                SharedPreferences preferences = PreferenceManager.getDefaultSharedPreferences(getApplicationContext()) ;
                resetActivityViews(preferences);
            }
        }
    }

    public void onBtnTakePictureClick(View view) {
        deletePhoto(currentPhotoPath);
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {

            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException e) {
                e.printStackTrace();
            }

            if (photoFile != null) {

                Uri photoURI = FileProvider.getUriForFile(this,
                        "sk.stigo.tensorflowliteocr.fileprovider",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            }
        }
    }

    public void onBtnRecognitionClick(View view) {
        SharedPreferences preferences = PreferenceManager.getDefaultSharedPreferences(getApplicationContext()) ;
        setLoadingSpinnerVisibility(View.VISIBLE);
        if (recognitionResult.getText() != null) {
            recognitionResult.setText(null);
        }
        Bitmap bitmap;

        try {
            if (preferences.getBoolean("useCamera", false)) {
                bitmap = getRotatedImg(BitmapFactory.decodeFile(currentPhotoPath, getBitMapOptions()), currentPhotoPath);

            } else {
                bitmap = getBitmapFromFile();
            }
        } catch (IOException e) {
            createErrorDialog("Unexpected exception occurred in onBtnRecognitionClick. Message: " + e.getMessage());
            return;
        }

        runTextRecognition(bitmap);
    }

    private Bitmap getBitmapFromFile() throws IOException {
        InputStream gaugeInputStream = getAssets().open("photo/" + selectedImageFile + ".jpg");
        return BitmapFactory.decodeStream(gaugeInputStream);
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image =  File.createTempFile(
                "cameraResult",  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    private void setPic() {
        Bitmap bitmap;
        try {
            bitmap = getRotatedImg(BitmapFactory.decodeFile(currentPhotoPath, getBitMapOptions()), currentPhotoPath);
            photoResultImageView.setImageBitmap(bitmap);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void runTextRecognition(Bitmap image) {
        Bitmap resultImage = ocrModelExecutor.run(image);


        List<String> results = ocrModelExecutor.getOcrResults();

        String resultStr = "No result";

        if (results.size() > 0) {
            resultStr = TextUtils.join("\n", results);
        }
        recognitionResult.setText(resultStr);

        photoResultImageView.setImageBitmap(resultImage);
        createResultDialog(resultStr);
        recognitionResult.setVisibility(View.VISIBLE);
        recognitionResultLabel.setVisibility(View.VISIBLE);
        setLoadingSpinnerVisibility(View.GONE);
    }

    private BitmapFactory.Options getBitMapOptions() {
        // Get the dimensions of the View
        int targetW = photoResultImageView.getWidth();
        int targetH = photoResultImageView.getHeight();

        // Get the dimensions of the bitmap
        BitmapFactory.Options bmOptions = new BitmapFactory.Options();
        bmOptions.inJustDecodeBounds = true;

        int photoW = bmOptions.outWidth;
        int photoH = bmOptions.outHeight;

        // Determine how much to scale down the image
        int scaleFactor = Math.min(photoW/targetW, photoH/targetH);

        // Decode the image file into a Bitmap sized to fill the View
        bmOptions.inJustDecodeBounds = false;
        bmOptions.inSampleSize = scaleFactor;
        bmOptions.inPurgeable = true;
        return bmOptions;
    }

    private void resetActivityViews(SharedPreferences preferences) {
        preferences.registerOnSharedPreferenceChangeListener(sharedPreferenceChangeListener);
        selectedImageFile = preferences.getString("selectFile", "g1");

        if (preferences.getBoolean("useCamera", false)) {
            photoResultImageView.setImageResource(R.drawable.photo_placeholder);
            btnTakePicture.setVisibility(View.VISIBLE);
            btnRecognize.setVisibility(View.GONE);
        } else {
            try {
                photoResultImageView.setImageBitmap(getBitmapFromFile());
            } catch (IOException e) {
                createErrorDialog("Unexpected exception occurred in onBtnRecognitionClick. Message: " + e.getMessage());
            }
            btnTakePicture.setVisibility(View.GONE);
            btnRecognize.setVisibility(View.VISIBLE);
        }
        recognitionResult.setText(null);
        recognitionResult.setVisibility(View.GONE);
        recognitionResultLabel.setVisibility(View.GONE);
    }

    private void setLoadingSpinnerVisibility(int visibility) {
        if (visibility == View.VISIBLE) {
            getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE,
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
        } else {
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
        }

        spinner.setVisibility(visibility);
    }

    private void createResultDialog(String message) {
        createAlertDialog(message, R.string.resultDialogTitle);
    }

    private void createErrorDialog(String message) {
        createAlertDialog(message, R.string.errorDialogTitle);
    }

    private void createAlertDialog(String message, int title) {
        new MaterialAlertDialogBuilder(MainActivity.this)
                .setMessage(message)
                .setTitle(title)
                .setNegativeButton(R.string.btnCancel, (dialog, which) -> dialog.dismiss())
                .show();
    }

}
