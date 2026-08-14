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
import java.util.PriorityQueue;
import java.util.Set;

final class ExecuTorchSsdDetector implements DetectionRunner {
    private static final int INPUT_SIZE = 640;
    private static final int CLASS_COUNT = 80;
    private static final int TORCHVISION_CLASS_COUNT = 91;
    private static final int PRE_NMS_TOP_K = 5000;
    private static final int MAXIMUM_DETECTIONS = 100;
    private static final float IOU_THRESHOLD = 0.50f;
    private static final double BBOX_XFORM_CLIP = Math.log(1000.0 / 16.0);
    private static final int[] COCO_CATEGORY_IDS = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    };
    private static final int[] FEATURE_MAP_SIZES = {80, 40, 20, 10, 5};
    private static final int[] ANCHOR_SIZES = {32, 64, 128, 256, 512};
    private static final float[] ASPECT_RATIOS = {0.5f, 1.0f, 2.0f};

    private final Module module;
    private final List<String> labels;
    private final List<Anchor> anchors;

    ExecuTorchSsdDetector(Context context, File modelFile) throws Exception {
        if (!modelFile.isFile() || modelFile.length() == 0) {
            throw new IllegalArgumentException("The imported ExecuTorch model is unavailable.");
        }
        labels = readLabels(context);
        anchors = buildAnchors();

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
            throw new IllegalArgumentException("Choose a valid image before detection.");
        }

        Tensor input = Tensor.fromBlob(
                preprocess(source),
                new long[]{1, 3, INPUT_SIZE, INPUT_SIZE}
        );
        long started = System.nanoTime();
        EValue[] outputs = module.execute("forward", EValue.from(input));
        long elapsedMilliseconds = Math.round((System.nanoTime() - started) / 1_000_000.0);

        if (outputs == null || outputs.length < 2) {
            throw new IllegalStateException("The SSD forward method must return two tensors.");
        }
        Tensor regression = requireTensor(outputs[0], 4, "bounding-box regression");
        Tensor logits = requireTensor(outputs[1], TORCHVISION_CLASS_COUNT, "class logits");
        int anchorCount = Math.toIntExact(regression.shape()[1]);
        if (anchorCount != anchors.size() || logits.shape()[1] != anchorCount) {
            throw new IllegalStateException(
                    "The SSD output anchor count does not match the 640 × 640 RetinaNet grid."
            );
        }

        List<Candidate> candidates = decode(
                regression.getDataAsFloatArray(),
                logits.getDataAsFloatArray(),
                confidenceThreshold,
                source
        );
        List<Detection> detections = nonMaximumSuppression(candidates);
        return new DetectionResult(elapsedMilliseconds, detections);
    }

    private List<Candidate> decode(float[] regression, float[] logits, float threshold,
                                   Bitmap source) {
        PriorityQueue<Candidate> bestCandidates = new PriorityQueue<>(
                Comparator.comparing(Candidate::score)
        );
        float scaleX = source.getWidth() / (float) INPUT_SIZE;
        float scaleY = source.getHeight() / (float) INPUT_SIZE;

        for (int anchorIndex = 0; anchorIndex < anchors.size(); anchorIndex++) {
            Anchor anchor = anchors.get(anchorIndex);
            int regressionOffset = anchorIndex * 4;
            Box decoded = decodeBox(
                    anchor,
                    regression[regressionOffset],
                    regression[regressionOffset + 1],
                    regression[regressionOffset + 2],
                    regression[regressionOffset + 3]
            );
            int logitsOffset = anchorIndex * TORCHVISION_CLASS_COUNT;
            for (int contiguousClass = 0; contiguousClass < CLASS_COUNT; contiguousClass++) {
                int categoryId = COCO_CATEGORY_IDS[contiguousClass];
                float score = sigmoid(logits[logitsOffset + categoryId]);
                if (score <= threshold) {
                    continue;
                }
                Candidate candidate = new Candidate(
                        decoded.left * scaleX,
                        decoded.top * scaleY,
                        decoded.right * scaleX,
                        decoded.bottom * scaleY,
                        score,
                        contiguousClass
                );
                bestCandidates.add(candidate);
                if (bestCandidates.size() > PRE_NMS_TOP_K) {
                    bestCandidates.poll();
                }
            }
        }
        return new ArrayList<>(bestCandidates);
    }

    private static Box decodeBox(Anchor anchor, float deltaX, float deltaY,
                                 float deltaWidth, float deltaHeight) {
        float width = anchor.right - anchor.left;
        float height = anchor.bottom - anchor.top;
        float centerX = anchor.left + 0.5f * width;
        float centerY = anchor.top + 0.5f * height;

        double decodedCenterX = deltaX * width + centerX;
        double decodedCenterY = deltaY * height + centerY;
        double decodedWidth = Math.exp(Math.min(deltaWidth, BBOX_XFORM_CLIP)) * width;
        double decodedHeight = Math.exp(Math.min(deltaHeight, BBOX_XFORM_CLIP)) * height;

        return new Box(
                clamp((float) (decodedCenterX - 0.5 * decodedWidth), 0.0f, INPUT_SIZE),
                clamp((float) (decodedCenterY - 0.5 * decodedHeight), 0.0f, INPUT_SIZE),
                clamp((float) (decodedCenterX + 0.5 * decodedWidth), 0.0f, INPUT_SIZE),
                clamp((float) (decodedCenterY + 0.5 * decodedHeight), 0.0f, INPUT_SIZE)
        );
    }

    private List<Detection> nonMaximumSuppression(List<Candidate> candidates) {
        candidates.removeIf(candidate -> candidate.right <= candidate.left
                || candidate.bottom <= candidate.top);
        candidates.sort(Comparator.comparing(Candidate::score).reversed());
        List<Candidate> kept = new ArrayList<>();
        for (Candidate candidate : candidates) {
            boolean suppressed = false;
            for (Candidate selected : kept) {
                if (candidate.classIndex == selected.classIndex
                        && intersectionOverUnion(candidate, selected) > IOU_THRESHOLD) {
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

    private static List<Anchor> buildAnchors() {
        List<Anchor> result = new ArrayList<>();
        double octaveScale = Math.pow(2.0, 1.0 / 3.0);
        for (int level = 0; level < FEATURE_MAP_SIZES.length; level++) {
            int gridSize = FEATURE_MAP_SIZES[level];
            int stride = INPUT_SIZE / gridSize;
            float[] scales = {
                    ANCHOR_SIZES[level],
                    (float) (ANCHOR_SIZES[level] * octaveScale),
                    (float) (ANCHOR_SIZES[level] * octaveScale * octaveScale)
            };
            List<Anchor> baseAnchors = new ArrayList<>(9);
            for (float aspectRatio : ASPECT_RATIOS) {
                double heightRatio = Math.sqrt(aspectRatio);
                double widthRatio = 1.0 / heightRatio;
                for (float scale : scales) {
                    float width = Math.round(widthRatio * scale);
                    float height = Math.round(heightRatio * scale);
                    baseAnchors.add(new Anchor(
                            -width / 2.0f,
                            -height / 2.0f,
                            width / 2.0f,
                            height / 2.0f
                    ));
                }
            }

            for (int y = 0; y < gridSize; y++) {
                float shiftY = y * stride;
                for (int x = 0; x < gridSize; x++) {
                    float shiftX = x * stride;
                    for (Anchor base : baseAnchors) {
                        result.add(new Anchor(
                                base.left + shiftX,
                                base.top + shiftY,
                                base.right + shiftX,
                                base.bottom + shiftY
                        ));
                    }
                }
            }
        }
        return List.copyOf(result);
    }

    private static Tensor requireTensor(EValue value, int trailingSize, String name) {
        if (!value.isTensor()) {
            throw new IllegalStateException("The SSD " + name + " output is not a tensor.");
        }
        Tensor tensor = value.toTensor();
        long[] shape = tensor.shape();
        if (tensor.dtype() != DType.FLOAT || shape.length != 3
                || shape[0] != 1 || shape[2] != trailingSize) {
            throw new IllegalStateException(
                    "Unexpected SSD " + name + " shape: " + Arrays.toString(shape)
            );
        }
        return tensor;
    }

    private float[] preprocess(Bitmap source) {
        Bitmap prepared = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(prepared);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(source, null, new RectF(0, 0, INPUT_SIZE, INPUT_SIZE), paint);

        int planeSize = INPUT_SIZE * INPUT_SIZE;
        int[] pixels = new int[planeSize];
        prepared.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        prepared.recycle();
        float[] data = new float[planeSize * 3];
        for (int index = 0; index < planeSize; index++) {
            int color = pixels[index];
            data[index] = Color.red(color) / 255.0f;
            data[planeSize + index] = Color.green(color) / 255.0f;
            data[2 * planeSize + index] = Color.blue(color) / 255.0f;
        }
        return data;
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

    private static float sigmoid(float value) {
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

    private record Anchor(float left, float top, float right, float bottom) {
    }

    private record Box(float left, float top, float right, float bottom) {
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
