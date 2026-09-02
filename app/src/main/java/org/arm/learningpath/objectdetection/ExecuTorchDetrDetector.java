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

final class ExecuTorchDetrDetector implements DetectionRunner {
    private static final int INPUT_SIZE = 640;
    private static final int CLASS_COUNT = 80;
    private static final int QUERY_COUNT = 300;
    private static final float IOU_THRESHOLD = 0.45f;
    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STANDARD_DEVIATION = {0.229f, 0.224f, 0.225f};
    private static final int[] COCO_CATEGORY_IDS = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    };

    private final Module module;
    private final Profile profile;
    private final List<String> labels;
    private final int[] categoryToContiguous = categoryMapping();

    ExecuTorchDetrDetector(Context context, File modelFile,
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
            throw new IllegalArgumentException("Choose a valid image before detection.");
        }

        Tensor input = Tensor.fromBlob(
                preprocess(source),
                new long[]{1, 3, INPUT_SIZE, INPUT_SIZE}
        );
        long started = System.nanoTime();
        EValue[] outputs = module.execute("forward", EValue.from(input));
        long elapsedMilliseconds = Math.round((System.nanoTime() - started) / 1_000_000.0);

        List<Candidate> candidates = switch (profile) {
            case RT_DETR -> decodeRtDetr(outputs, source, confidenceThreshold);
            case DEFORMABLE_DETR -> decodeDeformableDetr(
                    outputs,
                    source,
                    confidenceThreshold
            );
        };
        List<Detection> detections = nonMaximumSuppression(candidates);
        return new DetectionResult(elapsedMilliseconds, detections);
    }

    private List<Candidate> decodeRtDetr(EValue[] outputs, Bitmap source, float threshold) {
        Tensor output = requireTensor(outputs, 0, new long[]{1, QUERY_COUNT, 4 + CLASS_COUNT});
        float[] values = output.getDataAsFloatArray();
        List<Candidate> candidates = new ArrayList<>();
        int features = 4 + CLASS_COUNT;
        for (int query = 0; query < QUERY_COUNT; query++) {
            int offset = query * features;
            int bestClass = 0;
            float bestScore = 0.0f;
            for (int classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
                float score = probability(values[offset + 4 + classIndex]);
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = classIndex;
                }
            }
            if (bestScore >= threshold) {
                candidates.add(normalizedCandidate(
                        values[offset],
                        values[offset + 1],
                        values[offset + 2],
                        values[offset + 3],
                        bestScore,
                        bestClass,
                        source
                ));
            }
        }
        return candidates;
    }

    private List<Candidate> decodeDeformableDetr(EValue[] outputs, Bitmap source,
                                                  float threshold) {
        Tensor logitsTensor = requireTensor(
                outputs,
                0,
                new long[]{1, QUERY_COUNT, 91},
                new long[]{1, QUERY_COUNT, 92}
        );
        Tensor boxesTensor = requireTensor(outputs, 1, new long[]{1, QUERY_COUNT, 4});
        float[] logits = logitsTensor.getDataAsFloatArray();
        float[] boxes = boxesTensor.getDataAsFloatArray();
        List<Candidate> candidates = new ArrayList<>();
        int logitsCount = (int) logitsTensor.shape()[2];

        for (int query = 0; query < QUERY_COUNT; query++) {
            int logitsOffset = query * logitsCount;
            float maximum = -Float.MAX_VALUE;
            for (int classIndex = 0; classIndex < logitsCount; classIndex++) {
                maximum = Math.max(maximum, logits[logitsOffset + classIndex]);
            }
            double sum = 0.0;
            for (int classIndex = 0; classIndex < logitsCount; classIndex++) {
                sum += Math.exp(logits[logitsOffset + classIndex] - maximum);
            }

            int bestCategory = 0;
            float bestScore = 0.0f;
            for (int categoryId = 0; categoryId < logitsCount - 1; categoryId++) {
                float score = (float) (
                        Math.exp(logits[logitsOffset + categoryId] - maximum) / sum
                );
                if (score > bestScore) {
                    bestScore = score;
                    bestCategory = categoryId;
                }
            }
            int contiguousClass = bestCategory < categoryToContiguous.length
                    ? categoryToContiguous[bestCategory]
                    : -1;
            if (contiguousClass >= 0 && bestScore >= threshold) {
                int boxOffset = query * 4;
                candidates.add(normalizedCandidate(
                        boxes[boxOffset],
                        boxes[boxOffset + 1],
                        boxes[boxOffset + 2],
                        boxes[boxOffset + 3],
                        bestScore,
                        contiguousClass,
                        source
                ));
            }
        }
        return candidates;
    }

    private Candidate normalizedCandidate(float centerX, float centerY, float width,
                                          float height, float score, int classIndex,
                                          Bitmap source) {
        float left = (centerX - width / 2.0f) * source.getWidth();
        float top = (centerY - height / 2.0f) * source.getHeight();
        float right = (centerX + width / 2.0f) * source.getWidth();
        float bottom = (centerY + height / 2.0f) * source.getHeight();
        return new Candidate(
                clamp(left, 0.0f, source.getWidth()),
                clamp(top, 0.0f, source.getHeight()),
                clamp(right, 0.0f, source.getWidth()),
                clamp(bottom, 0.0f, source.getHeight()),
                score,
                classIndex
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
            float red = Color.red(color) / 255.0f;
            float green = Color.green(color) / 255.0f;
            float blue = Color.blue(color) / 255.0f;
            if (profile == Profile.DEFORMABLE_DETR) {
                red = (red - IMAGENET_MEAN[0]) / IMAGENET_STANDARD_DEVIATION[0];
                green = (green - IMAGENET_MEAN[1]) / IMAGENET_STANDARD_DEVIATION[1];
                blue = (blue - IMAGENET_MEAN[2]) / IMAGENET_STANDARD_DEVIATION[2];
            }
            data[index] = red;
            data[planeSize + index] = green;
            data[2 * planeSize + index] = blue;
        }
        return data;
    }

    private static Tensor requireTensor(EValue[] outputs, int index,
                                        long[]... expectedShapes) {
        if (outputs == null || outputs.length <= index || !outputs[index].isTensor()) {
            throw new IllegalStateException("The DETR forward method returned unexpected outputs.");
        }
        Tensor tensor = outputs[index].toTensor();
        boolean shapeMatches = false;
        for (long[] expectedShape : expectedShapes) {
            if (Arrays.equals(tensor.shape(), expectedShape)) {
                shapeMatches = true;
                break;
            }
        }
        if (tensor.dtype() != DType.FLOAT || !shapeMatches) {
            throw new IllegalStateException(
                    "Unexpected DETR tensor at output " + index + ": "
                            + Arrays.toString(tensor.shape())
            );
        }
        return tensor;
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

    private static int[] categoryMapping() {
        int[] mapping = new int[91];
        Arrays.fill(mapping, -1);
        for (int index = 0; index < COCO_CATEGORY_IDS.length; index++) {
            mapping[COCO_CATEGORY_IDS[index]] = index;
        }
        return mapping;
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
        RT_DETR,
        DEFORMABLE_DETR
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
