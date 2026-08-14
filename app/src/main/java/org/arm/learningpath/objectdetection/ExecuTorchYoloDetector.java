package org.arm.learningpath.objectdetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import org.json.JSONArray;
import org.pytorch.executorch.DType;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.MethodMetadata;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

final class ExecuTorchYoloDetector implements DetectionRunner {
    private static final int INPUT_SIZE = 640;
    private static final int CLASS_COUNT = 80;
    private static final int MAXIMUM_NMS_CANDIDATES = 3000;
    private static final int MAXIMUM_DETECTIONS = 300;

    private final Module module;
    private final Profile profile;
    private final List<String> labels;

    ExecuTorchYoloDetector(Context context, File modelFile,
                           Profile detectorProfile) throws Exception {
        if (!modelFile.isFile() || modelFile.length() == 0) {
            throw new IllegalArgumentException("The imported ExecuTorch model is unavailable.");
        }
        profile = detectorProfile;
        labels = readLabels(context);

        Module loadedModule = null;
        try {
            loadedModule = Module.load(modelFile.getAbsolutePath(), Module.LOAD_MODE_MMAP);
            validateModule(loadedModule);
            module = loadedModule;
        } catch (Exception exception) {
            if (loadedModule != null) {
                loadedModule.close();
            }
            throw exception;
        }
    }

    @Override
    public DetectionResult detect(Bitmap source, float confidenceThreshold) {
        if (source == null || source.isRecycled()
                || source.getWidth() < 1 || source.getHeight() < 1) {
            throw new IllegalArgumentException("Choose a valid photo before detection.");
        }
        if (confidenceThreshold < 0.0f || confidenceThreshold > 1.0f) {
            throw new IllegalArgumentException("The confidence threshold must be between 0 and 1.");
        }

        PreparedInput prepared = prepareInput(source);
        Tensor input = Tensor.fromBlob(
                prepared.tensorData(),
                new long[]{1, 3, INPUT_SIZE, INPUT_SIZE}
        );

        long started = System.nanoTime();
        EValue[] outputs = module.execute("forward", EValue.from(input));
        long elapsedMilliseconds = Math.round((System.nanoTime() - started) / 1_000_000.0);

        Tensor output = validateOutput(outputs);
        List<Candidate> candidates = decode(
                output.getDataAsFloatArray(),
                output.shape(),
                confidenceThreshold,
                prepared
        );
        List<Detection> detections = nonMaximumSuppression(candidates, profile.iouThreshold);
        return new DetectionResult(elapsedMilliseconds, detections);
    }

    private static void validateModule(Module module) {
        Set<String> methods = new HashSet<>(Arrays.asList(module.getMethods()));
        if (!methods.contains("forward")) {
            throw new IllegalArgumentException("The model is missing the forward method.");
        }
        module.loadMethod("forward");
        MethodMetadata metadata = module.getMethodMetadata("forward");
        Set<String> backends = new HashSet<>(Arrays.asList(metadata.getBackends()));
        if (!backends.contains("XnnpackBackend")) {
            throw new IllegalArgumentException(
                    "The model forward method does not declare the XNNPACK backend."
            );
        }
    }

    private Tensor validateOutput(EValue[] outputs) {
        if (outputs == null || outputs.length < 1 || !outputs[0].isTensor()) {
            throw new IllegalStateException("The forward method must return a tensor.");
        }
        Tensor output = outputs[0].toTensor();
        if (output.dtype() != DType.FLOAT) {
            throw new IllegalStateException("The detector output must use float32 values.");
        }
        long[] shape = output.shape();
        if (shape.length != 3 || shape[0] != 1 || !profile.matches(shape)) {
            throw new IllegalStateException(
                    "Unexpected " + profile.displayName + " output shape: "
                            + Arrays.toString(shape)
            );
        }
        return output;
    }

    private List<Candidate> decode(float[] output, long[] shape, float threshold,
                                   PreparedInput prepared) {
        return switch (profile) {
            case YOLO_V5 -> decodeYoloV5(output, shape, threshold, prepared);
            case YOLO_V8 -> decodeYoloV8(output, shape, threshold, prepared, false);
            case YOLO_V9 -> decodeYoloV8(output, shape, threshold, prepared, true);
        };
    }

    private List<Candidate> decodeYoloV5(float[] output, long[] shape, float threshold,
                                         PreparedInput prepared) {
        int candidates = Math.toIntExact(shape[1]);
        int features = Math.toIntExact(shape[2]);
        List<Candidate> decoded = new ArrayList<>();
        for (int candidate = 0; candidate < candidates; candidate++) {
            int offset = candidate * features;
            float objectness = probability(output[offset + 4]);
            int bestClass = 0;
            float bestClassScore = 0.0f;
            for (int classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
                float classScore = probability(output[offset + 5 + classIndex]);
                if (classScore > bestClassScore) {
                    bestClassScore = classScore;
                    bestClass = classIndex;
                }
            }
            float score = objectness * bestClassScore;
            if (score >= threshold) {
                decoded.add(candidate(
                        output[offset],
                        output[offset + 1],
                        output[offset + 2],
                        output[offset + 3],
                        score,
                        bestClass,
                        prepared
                ));
            }
        }
        return decoded;
    }

    private List<Candidate> decodeYoloV8(float[] output, long[] shape, float threshold,
                                         PreparedInput prepared, boolean multiLabel) {
        boolean featureFirst = shape[1] == 4 + CLASS_COUNT;
        int candidates = Math.toIntExact(featureFirst ? shape[2] : shape[1]);
        int features = Math.toIntExact(featureFirst ? shape[1] : shape[2]);
        List<Candidate> decoded = new ArrayList<>();
        for (int candidate = 0; candidate < candidates; candidate++) {
            float centerX = value(output, candidate, 0, candidates, features, featureFirst);
            float centerY = value(output, candidate, 1, candidates, features, featureFirst);
            float width = value(output, candidate, 2, candidates, features, featureFirst);
            float height = value(output, candidate, 3, candidates, features, featureFirst);

            if (multiLabel) {
                for (int classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
                    float score = probability(value(
                            output,
                            candidate,
                            4 + classIndex,
                            candidates,
                            features,
                            featureFirst
                    ));
                    if (score >= threshold) {
                        decoded.add(candidate(
                                centerX,
                                centerY,
                                width,
                                height,
                                score,
                                classIndex,
                                prepared
                        ));
                    }
                }
            } else {
                int bestClass = 0;
                float bestScore = 0.0f;
                for (int classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
                    float score = probability(value(
                            output,
                            candidate,
                            4 + classIndex,
                            candidates,
                            features,
                            featureFirst
                    ));
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = classIndex;
                    }
                }
                if (bestScore >= threshold) {
                    decoded.add(candidate(
                            centerX,
                            centerY,
                            width,
                            height,
                            bestScore,
                            bestClass,
                            prepared
                    ));
                }
            }
        }
        return decoded;
    }

    private Candidate candidate(float centerX, float centerY, float width, float height,
                                float score, int classIndex, PreparedInput prepared) {
        float left = centerX - width / 2.0f;
        float top = centerY - height / 2.0f;
        float right = centerX + width / 2.0f;
        float bottom = centerY + height / 2.0f;

        if (prepared.letterboxed()) {
            left = (left - prepared.padX()) / prepared.scale();
            right = (right - prepared.padX()) / prepared.scale();
            top = (top - prepared.padY()) / prepared.scale();
            bottom = (bottom - prepared.padY()) / prepared.scale();
        } else {
            left *= prepared.originalWidth() / (float) INPUT_SIZE;
            right *= prepared.originalWidth() / (float) INPUT_SIZE;
            top *= prepared.originalHeight() / (float) INPUT_SIZE;
            bottom *= prepared.originalHeight() / (float) INPUT_SIZE;
        }

        return new Candidate(
                clamp(left, 0.0f, prepared.originalWidth()),
                clamp(top, 0.0f, prepared.originalHeight()),
                clamp(right, 0.0f, prepared.originalWidth()),
                clamp(bottom, 0.0f, prepared.originalHeight()),
                score,
                classIndex
        );
    }

    private List<Detection> nonMaximumSuppression(List<Candidate> candidates, float iouThreshold) {
        candidates.removeIf(candidate -> candidate.right <= candidate.left
                || candidate.bottom <= candidate.top);
        candidates.sort(Comparator.comparing(Candidate::score).reversed());
        if (candidates.size() > MAXIMUM_NMS_CANDIDATES) {
            candidates = new ArrayList<>(candidates.subList(0, MAXIMUM_NMS_CANDIDATES));
        }

        List<Candidate> kept = new ArrayList<>();
        for (Candidate candidate : candidates) {
            boolean suppressed = false;
            for (Candidate selected : kept) {
                if (candidate.classIndex == selected.classIndex
                        && intersectionOverUnion(candidate, selected) > iouThreshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept.add(candidate);
                if (kept.size() == MAXIMUM_DETECTIONS) {
                    break;
                }
            }
        }

        List<Detection> detections = new ArrayList<>(kept.size());
        for (Candidate candidate : kept) {
            detections.add(new Detection(
                    candidate.left,
                    candidate.top,
                    candidate.right,
                    candidate.bottom,
                    candidate.score,
                    candidate.classIndex,
                    labels.get(candidate.classIndex)
            ));
        }
        return detections;
    }

    private PreparedInput prepareInput(Bitmap source) {
        boolean letterbox = profile == Profile.YOLO_V9;
        Bitmap prepared = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(prepared);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);

        float scale;
        float padX;
        float padY;
        if (letterbox) {
            canvas.drawColor(Color.rgb(114, 114, 114));
            scale = Math.min(
                    INPUT_SIZE / (float) source.getWidth(),
                    INPUT_SIZE / (float) source.getHeight()
            );
            float width = source.getWidth() * scale;
            float height = source.getHeight() * scale;
            padX = (INPUT_SIZE - width) / 2.0f;
            padY = (INPUT_SIZE - height) / 2.0f;
            canvas.drawBitmap(source, null, new RectF(padX, padY, padX + width, padY + height), paint);
        } else {
            scale = 1.0f;
            padX = 0.0f;
            padY = 0.0f;
            canvas.drawBitmap(source, null, new RectF(0, 0, INPUT_SIZE, INPUT_SIZE), paint);
        }

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        prepared.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        prepared.recycle();

        int planeSize = INPUT_SIZE * INPUT_SIZE;
        float[] data = new float[planeSize * 3];
        for (int index = 0; index < planeSize; index++) {
            int color = pixels[index];
            data[index] = Color.red(color) / 255.0f;
            data[planeSize + index] = Color.green(color) / 255.0f;
            data[2 * planeSize + index] = Color.blue(color) / 255.0f;
        }
        return new PreparedInput(
                data,
                source.getWidth(),
                source.getHeight(),
                letterbox,
                scale,
                padX,
                padY
        );
    }

    private static float value(float[] output, int candidate, int feature,
                               int candidates, int features, boolean featureFirst) {
        return featureFirst
                ? output[feature * candidates + candidate]
                : output[candidate * features + feature];
    }

    private static float probability(float value) {
        if (value >= 0.0f && value <= 1.0f) {
            return value;
        }
        float clipped = clamp(value, -50.0f, 50.0f);
        return (float) (1.0 / (1.0 + Math.exp(-clipped)));
    }

    private static float intersectionOverUnion(Candidate first, Candidate second) {
        float left = Math.max(first.left, second.left);
        float top = Math.max(first.top, second.top);
        float right = Math.min(first.right, second.right);
        float bottom = Math.min(first.bottom, second.bottom);
        float intersection = Math.max(0.0f, right - left) * Math.max(0.0f, bottom - top);
        float firstArea = (first.right - first.left) * (first.bottom - first.top);
        float secondArea = (second.right - second.left) * (second.bottom - second.top);
        float union = firstArea + secondArea - intersection;
        return union <= 0.0f ? 0.0f : intersection / union;
    }

    private static List<String> readLabels(Context context) throws Exception {
        StringBuilder text = new StringBuilder();
        try (InputStream input = context.getAssets().open("coco_labels.json");
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(input, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                text.append(line);
            }
        }
        JSONArray array = new JSONArray(text.toString());
        if (array.length() != CLASS_COUNT) {
            throw new IllegalArgumentException("The COCO label file must contain 80 labels.");
        }
        List<String> result = new ArrayList<>(CLASS_COUNT);
        for (int index = 0; index < array.length(); index++) {
            result.add(array.getString(index));
        }
        return List.copyOf(result);
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    @Override
    public void close() {
        module.close();
    }

    enum Profile {
        YOLO_V5("YOLOv5", 0.45f),
        YOLO_V8("YOLOv8", 0.45f),
        YOLO_V9("YOLOv9", 0.70f);

        private final String displayName;
        private final float iouThreshold;

        Profile(String displayName, float iouThreshold) {
            this.displayName = displayName;
            this.iouThreshold = iouThreshold;
        }

        boolean matches(long[] shape) {
            return switch (this) {
                case YOLO_V5 -> shape[2] == 85;
                case YOLO_V8, YOLO_V9 -> shape[1] == 84 || shape[2] == 84;
            };
        }
    }

    private record PreparedInput(
            float[] tensorData,
            int originalWidth,
            int originalHeight,
            boolean letterboxed,
            float scale,
            float padX,
            float padY
    ) {
    }

    private record Candidate(
            float left,
            float top,
            float right,
            float bottom,
            float score,
            int classIndex
    ) {
    }
}
