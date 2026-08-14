package org.arm.learningpath.objectdetection;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.activity.ComponentActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends ComponentActivity {
    private static final int IMPORT_MODEL_REQUEST = 1001;
    private static final int OPEN_IMAGE_REQUEST = 1002;
    private static final int CAMERA_PERMISSION_REQUEST = 1003;
    private static final int DEFAULT_CONFIDENCE_PERCENT = 75;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Map<String, View> modeCards = new LinkedHashMap<>();
    private File modelsDirectory;
    private List<DetectionAdapter> adapters;
    private DetectionAdapter selectedAdapter;
    private ModelImporter.State activeState;
    private DetectionRunner runner;
    private String runnerDescriptorId;
    private Bitmap selectedBitmap;
    private LinearLayout modeSelector;
    private TextView modelStatus;
    private TextView modelExplanation;
    private TextView confidenceValue;
    private TextView results;
    private Button importModel;
    private Button chooseImage;
    private Button cameraButton;
    private Button runDetection;
    private SeekBar confidenceThreshold;
    private DetectionOverlayView imagePreview;
    private DetectionOverlayView cameraOverlay;
    private PreviewView cameraPreview;
    private ProcessCameraProvider cameraProvider;
    private boolean cameraStarting;
    private volatile boolean cameraActive;
    private volatile int confidencePercent = DEFAULT_CONFIDENCE_PERCENT;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        modelsDirectory = new File(getFilesDir(), "detection-models");
        adapters = AdapterRegistry.all();
        if (adapters.isEmpty()) {
            throw new IllegalStateException("The application does not contain any adapters.");
        }

        modeSelector = findViewById(R.id.mode_selector);
        modelStatus = findViewById(R.id.model_status);
        modelExplanation = findViewById(R.id.model_explanation);
        confidenceValue = findViewById(R.id.confidence_value);
        results = findViewById(R.id.results);
        importModel = findViewById(R.id.import_model);
        chooseImage = findViewById(R.id.choose_image);
        cameraButton = findViewById(R.id.camera_button);
        runDetection = findViewById(R.id.run_detection);
        confidenceThreshold = findViewById(R.id.confidence_threshold);
        imagePreview = findViewById(R.id.image_preview);
        cameraOverlay = findViewById(R.id.camera_overlay);
        cameraPreview = findViewById(R.id.camera_preview);

        importModel.setOnClickListener(view -> openModelPicker());
        chooseImage.setOnClickListener(view -> openImagePicker());
        cameraButton.setOnClickListener(view -> toggleCamera());
        runDetection.setOnClickListener(view -> runDetection());
        confidenceThreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                confidencePercent = progress;
                confidenceValue.setText(getString(R.string.confidence_value, progress));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        createModeCards();
        activeState = ModelImporter.activeState(modelsDirectory);
        DetectionAdapter initialAdapter = activeState == null
                ? adapters.get(0)
                : AdapterRegistry.forId(activeState.adapterId());
        selectAdapter(initialAdapter == null ? adapters.get(0) : initialAdapter, false);
    }

    private void createModeCards() {
        LayoutInflater inflater = LayoutInflater.from(this);
        for (DetectionAdapter adapter : adapters) {
            View card = inflater.inflate(R.layout.view_mode_card, modeSelector, false);
            AdapterDefinition definition = adapter.definition();
            ((TextView) card.findViewById(R.id.mode_title)).setText(definition.titleResource());
            ((TextView) card.findViewById(R.id.mode_description)).setText(
                    definition.descriptionResource()
            );
            LinearLayout.LayoutParams parameters = adapters.size() == 1
                    ? new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
            )
                    : new LinearLayout.LayoutParams(dpToPixels(190), ViewGroup.LayoutParams.WRAP_CONTENT);
            parameters.setMarginEnd(dpToPixels(12));
            card.setLayoutParams(parameters);
            card.setOnClickListener(view -> selectAdapter(adapter, true));
            modeSelector.addView(card);
            modeCards.put(definition.id(), card);
        }
    }

    private void selectAdapter(DetectionAdapter adapter, boolean activateStoredModel) {
        stopCamera(false);
        closeRunner();
        selectedAdapter = adapter;
        ModelImporter.State stored = ModelImporter.stateForAdapter(
                modelsDirectory,
                adapter.definition().id()
        );
        activeState = stored;
        confidenceThreshold.setProgress(
                stored == null
                        ? DEFAULT_CONFIDENCE_PERCENT
                        : stored.descriptor().defaultConfidencePercent()
        );
        if (activateStoredModel && stored != null) {
            try {
                ModelImporter.activate(modelsDirectory, stored);
            } catch (Exception exception) {
                showError(getString(R.string.model_import_failed, exception.getMessage()));
                return;
            }
        }
        applyModeState();
    }

    private void applyModeState() {
        for (Map.Entry<String, View> entry : modeCards.entrySet()) {
            entry.getValue().setBackgroundResource(
                    entry.getKey().equals(selectedAdapter.definition().id())
                            ? R.drawable.bg_mode_selected
                            : R.drawable.bg_mode_unselected
            );
        }
        AdapterDefinition definition = selectedAdapter.definition();
        importModel.setText(definition.importButtonResource());
        if (activeState == null) {
            modelStatus.setText(R.string.model_not_imported);
            modelExplanation.setText(definition.missingModelResource());
        } else {
            modelStatus.setText(activeState.descriptor().displayLabel());
            modelExplanation.setText(definition.readyModelResource());
        }
        updateControls(false);
    }

    private void openModelPicker() {
        stopCamera(false);
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/octet-stream");
        startActivityForResult(intent, IMPORT_MODEL_REQUEST);
    }

    private void openImagePicker() {
        stopCamera(false);
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, OPEN_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        if (requestCode == IMPORT_MODEL_REQUEST) {
            importSelectedModel(data.getData());
        } else if (requestCode == OPEN_IMAGE_REQUEST) {
            loadSelectedImage(data.getData());
        }
    }

    private void importSelectedModel(Uri uri) {
        setBusy(getString(R.string.importing_model));
        executor.execute(() -> {
            try {
                closeRunner();
                ModelImporter.ImportResult imported = ModelImporter.importModel(
                        getApplicationContext(),
                        uri,
                        modelsDirectory
                );
                DetectionAdapter importedAdapter = AdapterRegistry.forId(
                        imported.state().adapterId()
                );
                if (importedAdapter == null) {
                    throw new IllegalStateException("The imported model adapter is unavailable.");
                }
                runOnUiThread(() -> {
                    selectedAdapter = importedAdapter;
                    activeState = imported.state();
                    confidenceThreshold.setProgress(
                            activeState.descriptor().defaultConfidencePercent()
                    );
                    applyModeState();
                    results.setText(R.string.model_ready_result);
                    setBusy(null);
                });
            } catch (Exception exception) {
                runOnUiThread(() -> showError(
                        getString(R.string.model_import_failed, exception.getMessage())
                ));
            }
        });
    }

    private void loadSelectedImage(Uri uri) {
        try (InputStream input = getContentResolver().openInputStream(uri)) {
            Bitmap bitmap = BitmapFactory.decodeStream(input);
            if (bitmap == null) {
                throw new IOException("Android could not decode the selected image.");
            }
            if (selectedBitmap != null && !selectedBitmap.isRecycled()) {
                selectedBitmap.recycle();
            }
            selectedBitmap = bitmap;
            cameraPreview.setVisibility(View.GONE);
            cameraOverlay.setVisibility(View.GONE);
            imagePreview.setVisibility(View.VISIBLE);
            imagePreview.setBitmap(bitmap);
            results.setText(R.string.image_ready);
            updateControls(false);
        } catch (IOException exception) {
            showError(getString(R.string.image_open_failed, exception.getMessage()));
        }
    }

    private void toggleCamera() {
        if (cameraActive || cameraStarting) {
            stopCamera(true);
            return;
        }
        if (activeState == null
                || !activeState.adapterId().equals(selectedAdapter.definition().id())) {
            showError(getString(R.string.import_before_camera));
            return;
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST
            );
            return;
        }
        startCamera();
    }

    private void startCamera() {
        cameraStarting = true;
        results.setText(R.string.starting_camera);
        updateControls(false);
        ListenableFuture<ProcessCameraProvider> providerFuture =
                ProcessCameraProvider.getInstance(this);
        providerFuture.addListener(() -> {
            try {
                cameraProvider = providerFuture.get();
                if (!cameraStarting) {
                    return;
                }
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(cameraPreview.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build();
                analysis.setAnalyzer(executor, this::analyzeCameraFrame);

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis
                );
                imagePreview.setVisibility(View.GONE);
                cameraPreview.setVisibility(View.VISIBLE);
                cameraOverlay.setVisibility(View.VISIBLE);
                cameraStarting = false;
                cameraActive = true;
                cameraButton.setText(R.string.stop_camera);
                results.setText(R.string.camera_running);
                updateControls(false);
            } catch (Exception exception) {
                cameraStarting = false;
                cameraActive = false;
                showError(getString(R.string.camera_start_failed, exception.getMessage()));
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyzeCameraFrame(ImageProxy image) {
        if (!cameraActive) {
            image.close();
            return;
        }
        Bitmap frame = null;
        try {
            frame = imageToBitmap(image);
            DetectionResult detectionResult = getRunner().detect(
                    frame,
                    confidencePercent / 100.0f
            );
            int frameWidth = frame.getWidth();
            int frameHeight = frame.getHeight();
            runOnUiThread(() -> {
                if (!cameraActive) {
                    return;
                }
                cameraOverlay.setCameraFrameSize(frameWidth, frameHeight);
                cameraOverlay.setDetections(detectionResult.detections());
                results.setText(detectionResult.summary());
            });
        } catch (Exception exception) {
            runOnUiThread(() -> {
                if (cameraActive) {
                    stopCamera(false);
                    showError(getString(R.string.inference_failed, exception.getMessage()));
                }
            });
        } finally {
            if (frame != null && !frame.isRecycled()) {
                frame.recycle();
            }
            image.close();
        }
    }

    private Bitmap imageToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        buffer.rewind();
        int width = image.getWidth();
        int height = image.getHeight();
        int rowPadding = plane.getRowStride() - plane.getPixelStride() * width;
        int paddedWidth = width + rowPadding / plane.getPixelStride();
        Bitmap padded = Bitmap.createBitmap(paddedWidth, height, Bitmap.Config.ARGB_8888);
        padded.copyPixelsFromBuffer(buffer);
        Bitmap cropped = Bitmap.createBitmap(padded, 0, 0, width, height);
        if (cropped != padded) {
            padded.recycle();
        }

        int rotationDegrees = image.getImageInfo().getRotationDegrees();
        if (rotationDegrees == 0) {
            return cropped;
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(rotationDegrees);
        Bitmap rotated = Bitmap.createBitmap(
                cropped,
                0,
                0,
                cropped.getWidth(),
                cropped.getHeight(),
                matrix,
                true
        );
        if (rotated != cropped) {
            cropped.recycle();
        }
        return rotated;
    }

    private void stopCamera(boolean showMessage) {
        cameraStarting = false;
        cameraActive = false;
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        cameraOverlay.clear();
        cameraOverlay.setVisibility(View.GONE);
        cameraPreview.setVisibility(View.GONE);
        imagePreview.setVisibility(View.VISIBLE);
        cameraButton.setText(R.string.start_camera);
        if (showMessage) {
            results.setText(selectedBitmap == null
                    ? R.string.camera_stopped
                    : R.string.image_ready);
        }
        updateControls(false);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                           int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != CAMERA_PERMISSION_REQUEST) {
            return;
        }
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            showError(getString(R.string.camera_permission_denied));
        }
    }

    private void runDetection() {
        if (activeState == null
                || !activeState.adapterId().equals(selectedAdapter.definition().id())) {
            showError(getString(R.string.import_before_running));
            return;
        }
        if (selectedBitmap == null) {
            showError(getString(R.string.choose_before_running));
            return;
        }
        float threshold = confidenceThreshold.getProgress() / 100.0f;
        setBusy(getString(selectedAdapter.definition().runningResource()));
        executor.execute(() -> {
            try {
                DetectionResult detectionResult = getRunner().detect(selectedBitmap, threshold);
                runOnUiThread(() -> {
                    imagePreview.setDetections(detectionResult.detections());
                    results.setText(detectionResult.summary());
                    setBusy(null);
                });
            } catch (Exception exception) {
                runOnUiThread(() -> showError(
                        getString(R.string.inference_failed, exception.getMessage())
                ));
            }
        });
    }

    private DetectionRunner getRunner() throws Exception {
        String descriptorId = activeState.descriptor().id();
        if (runner != null && descriptorId.equals(runnerDescriptorId)) {
            return runner;
        }
        closeRunner();
        runner = selectedAdapter.createRunner(
                getApplicationContext(),
                ModelImporter.modelFile(modelsDirectory, activeState.descriptor()),
                activeState.descriptor()
        );
        runnerDescriptorId = descriptorId;
        return runner;
    }

    private void setBusy(String message) {
        boolean busy = message != null;
        if (busy) {
            results.setText(message);
        }
        updateControls(busy);
    }

    private void updateControls(boolean busy) {
        boolean cameraInUse = cameraActive || cameraStarting;
        importModel.setEnabled(!busy && !cameraInUse);
        chooseImage.setEnabled(!busy && !cameraInUse);
        cameraButton.setEnabled(!busy && activeState != null);
        confidenceThreshold.setEnabled(!busy);
        runDetection.setEnabled(
                !busy && !cameraInUse && activeState != null && selectedBitmap != null
        );
    }

    private void showError(String message) {
        results.setText(message);
        updateControls(false);
    }

    private void closeRunner() {
        if (runner != null) {
            runner.close();
            runner = null;
            runnerDescriptorId = null;
        }
    }

    private int dpToPixels(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override
    protected void onDestroy() {
        stopCamera(false);
        closeRunner();
        executor.shutdownNow();
        if (selectedBitmap != null && !selectedBitmap.isRecycled()) {
            selectedBitmap.recycle();
        }
        super.onDestroy();
    }
}
