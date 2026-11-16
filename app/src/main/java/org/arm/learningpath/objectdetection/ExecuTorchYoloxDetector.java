package org.arm.learningpath.objectdetection;

import android.content.Context;
import android.graphics.Bitmap;

import org.json.JSONArray;
import org.pytorch.executorch.DType;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** YOLOX-S export with separate regression, objectness and class heads per stride. */
final class ExecuTorchYoloxDetector implements DetectionRunner {
    private static final int INPUT_SIZE = 640;
    private static final int CLASS_COUNT = 80;
    private static final int[] STRIDES = {8, 16, 32};
    private static final int MAXIMUM_NMS_CANDIDATES = 3000;
    private static final int MAXIMUM_DETECTIONS = 300;
    private static final float NMS_IOU = 0.65f;

    private final Module module;
    private final List<String> labels;

    ExecuTorchYoloxDetector(Context context, File modelFile) throws Exception {
        if (!modelFile.isFile() || modelFile.length() == 0) {
            throw new IllegalArgumentException("The imported ExecuTorch model is unavailable.");
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                context.getAssets().open("coco_labels.json"), StandardCharsets.UTF_8))) {
            StringBuilder text = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) text.append(line);
            JSONArray array = new JSONArray(text.toString());
            if (array.length() != CLASS_COUNT) {
                throw new IllegalArgumentException("The COCO label file must contain 80 labels.");
            }
            List<String> names = new ArrayList<>(CLASS_COUNT);
            for (int index = 0; index < CLASS_COUNT; index++) names.add(array.getString(index));
            labels = List.copyOf(names);
        }
        Module loaded = Module.load(modelFile.getAbsolutePath(), Module.LOAD_MODE_MMAP);
        try {
            if (!Arrays.asList(loaded.getMethods()).contains("forward")) {
                throw new IllegalArgumentException("The model is missing the forward method.");
            }
            loaded.loadMethod("forward");
            if (!Arrays.asList(loaded.getMethodMetadata("forward").getBackends())
                    .contains("XnnpackBackend")) {
                throw new IllegalArgumentException("The model does not declare the XNNPACK backend.");
            }
            module = loaded;
        } catch (Exception exception) {
            loaded.close();
            throw exception;
        }
    }

    @Override
    public DetectionResult detect(Bitmap source, float confidenceThreshold) {
        if (source == null || source.isRecycled()
                || source.getWidth() < 1 || source.getHeight() < 1) {
            throw new IllegalArgumentException("Choose a valid photo before detection.");
        }
        if (!Float.isFinite(confidenceThreshold)
                || confidenceThreshold < 0.0f || confidenceThreshold > 1.0f) {
            throw new IllegalArgumentException("The confidence threshold must be between 0 and 1.");
        }
        PreparedInput prepared = prepareInput(source);
        Tensor input = Tensor.fromBlob(prepared.tensorData(), new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
        long started = System.nanoTime();
        EValue[] outputs = module.execute("forward", EValue.from(input));
        long elapsed = Math.round((System.nanoTime() - started) / 1_000_000.0);
        float[][] heads = validateOutputs(outputs);
        return new DetectionResult(elapsed, decode(heads, source.getWidth(), source.getHeight(),
                prepared.scale(), confidenceThreshold, labels));
    }

    static PreparedInput prepareInput(Bitmap source) {
        float scale = Math.min(INPUT_SIZE / (float) source.getWidth(),
                INPUT_SIZE / (float) source.getHeight());
        int width = Math.max(1, (int) (source.getWidth() * scale));
        int height = Math.max(1, (int) (source.getHeight() * scale));
        int sourceWidth = source.getWidth();
        int sourceHeight = source.getHeight();
        int[] pixels = new int[sourceWidth * sourceHeight];
        source.getPixels(pixels, 0, sourceWidth, 0, 0, sourceWidth, sourceHeight);
        int plane = INPUT_SIZE * INPUT_SIZE;
        float[] data = new float[plane * 3];
        Arrays.fill(data, 114f);
        // Half-pixel bilinear coordinates match the upstream OpenCV resize convention.
        // Pixel rounding may differ by one intensity level from OpenCV's fixed-point path.
        for (int y = 0; y < height; y++) {
            double sourceY = (y + 0.5) * sourceHeight / height - 0.5;
            int top = (int) Math.floor(sourceY);
            double dy = sourceY - top;
            int row0 = Math.max(0, Math.min(sourceHeight - 1, top)) * sourceWidth;
            int row1 = Math.max(0, Math.min(sourceHeight - 1, top + 1)) * sourceWidth;
            for (int x = 0; x < width; x++) {
                double sourceX = (x + 0.5) * sourceWidth / width - 0.5;
                int left = (int) Math.floor(sourceX);
                double dx = sourceX - left;
                int column0 = Math.max(0, Math.min(sourceWidth - 1, left));
                int column1 = Math.max(0, Math.min(sourceWidth - 1, left + 1));
                int p00 = pixels[row0 + column0];
                int p01 = pixels[row0 + column1];
                int p10 = pixels[row1 + column0];
                int p11 = pixels[row1 + column1];
                int offset = y * INPUT_SIZE + x;
                for (int channel = 0; channel < 3; channel++) {
                    // Low byte first yields BGR values in [0, 255], without normalization.
                    int shift = channel * 8;
                    double first = ((p00 >> shift) & 255) * (1 - dx) + ((p01 >> shift) & 255) * dx;
                    double second = ((p10 >> shift) & 255) * (1 - dx) + ((p11 >> shift) & 255) * dx;
                    data[channel * plane + offset] = Math.round(first * (1 - dy) + second * dy);
                }
            }
        }
        return new PreparedInput(data, scale);
    }

    private static float[][] validateOutputs(EValue[] outputs) {
        if (outputs == null || outputs.length != 9) {
            throw new IllegalStateException("YOLOX-S must return nine prediction tensors.");
        }
        float[][] heads = new float[9][];
        int[] channels = {4, 1, CLASS_COUNT};
        for (int level = 0; level < STRIDES.length; level++) {
            int size = INPUT_SIZE / STRIDES[level];
            for (int branch = 0; branch < channels.length; branch++) {
                int index = level * 3 + branch;
                if (!outputs[index].isTensor()) {
                    throw new IllegalStateException("YOLOX-S prediction must be a tensor.");
                }
                Tensor tensor = outputs[index].toTensor();
                if (tensor.dtype() != DType.FLOAT
                        || !Arrays.equals(tensor.shape(), new long[]{1, channels[branch], size, size})) {
                    throw new IllegalStateException("Unexpected YOLOX-S prediction tensor " + index
                            + ": " + Arrays.toString(tensor.shape()));
                }
                heads[index] = tensor.getDataAsFloatArray();
                for (float value : heads[index]) {
                    if (!Float.isFinite(value)) {
                        throw new IllegalStateException("YOLOX-S prediction contains a non-finite value.");
                    }
                }
            }
        }
        return heads;
    }

    static List<Detection> decode(float[][] heads, int width, int height, float scale,
                                  float threshold, List<String> labels) {
        List<Detection> candidates = new ArrayList<>();
        for (int level = 0; level < STRIDES.length; level++) {
            int stride = STRIDES[level];
            int size = INPUT_SIZE / stride;
            int plane = size * size;
            float[] boxes = heads[level * 3];
            float[] objectness = heads[level * 3 + 1];
            float[] classes = heads[level * 3 + 2];
            for (int cell = 0; cell < plane; cell++) {
                float objectScore = sigmoid(objectness[cell]);
                if (objectScore < threshold) continue;
                int bestClass = 0;
                float bestLogit = classes[cell];
                for (int index = 1; index < CLASS_COUNT; index++) {
                    float logit = classes[index * plane + cell];
                    if (logit > bestLogit) {
                        bestLogit = logit;
                        bestClass = index;
                    }
                }
                float score = objectScore * sigmoid(bestLogit);
                if (score < threshold) continue;
                float centerX = (boxes[cell] + cell % size) * stride;
                float centerY = (boxes[plane + cell] + cell / size) * stride;
                float boxWidth = (float) Math.exp(boxes[2 * plane + cell]) * stride;
                float boxHeight = (float) Math.exp(boxes[3 * plane + cell]) * stride;
                if (!Float.isFinite(centerX) || !Float.isFinite(centerY)
                        || !Float.isFinite(boxWidth) || !Float.isFinite(boxHeight)) continue;
                float left = clamp((centerX - boxWidth / 2) / scale, width);
                float top = clamp((centerY - boxHeight / 2) / scale, height);
                float right = clamp((centerX + boxWidth / 2) / scale, width);
                float bottom = clamp((centerY + boxHeight / 2) / scale, height);
                if (right > left && bottom > top) {
                    candidates.add(new Detection(left, top, right, bottom, score,
                            bestClass, labels.get(bestClass)));
                }
            }
        }
        candidates.sort(Comparator.comparing(Detection::score).reversed());
        if (candidates.size() > MAXIMUM_NMS_CANDIDATES) {
            candidates = new ArrayList<>(candidates.subList(0, MAXIMUM_NMS_CANDIDATES));
        }
        List<Detection> kept = new ArrayList<>();
        for (Detection candidate : candidates) {
            boolean duplicate = false;
            for (Detection previous : kept) {
                if (candidate.classIndex() == previous.classIndex() && iou(candidate, previous) > NMS_IOU) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                kept.add(candidate);
                if (kept.size() == MAXIMUM_DETECTIONS) break;
            }
        }
        return kept;
    }

    private static float sigmoid(float logit) {
        return (float) (1.0 / (1.0 + Math.exp(-logit)));
    }

    private static float clamp(float value, int maximum) {
        return Math.max(0, Math.min(maximum, value));
    }

    private static float iou(Detection a, Detection b) {
        float intersection = Math.max(0, Math.min(a.right(), b.right()) - Math.max(a.left(), b.left()))
                * Math.max(0, Math.min(a.bottom(), b.bottom()) - Math.max(a.top(), b.top()));
        float union = (a.right() - a.left()) * (a.bottom() - a.top())
                + (b.right() - b.left()) * (b.bottom() - b.top()) - intersection;
        return union > 0 ? intersection / union : 0;
    }

    @Override
    public void close() {
        module.close();
    }

    record PreparedInput(float[] tensorData, float scale) { }
}
