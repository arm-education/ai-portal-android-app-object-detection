package org.arm.learningpath.objectdetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.TensorBuffer;
import com.google.ai.edge.litert.TensorType;

import org.json.JSONArray;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

final class LiteRtYoloDetector implements DetectionRunner {
    private static final int INPUT_SIZE = 640;
    private static final int CLASS_COUNT = 80;
    private static final int YOLO_11_CANDIDATE_COUNT = 8400;
    private static final int YOLO_11_FEATURE_COUNT = 4 + CLASS_COUNT;
    private static final int YOLO_26_CANDIDATE_COUNT = 300;
    private static final int YOLO_26_FEATURE_COUNT = 6;
    private static final int MAXIMUM_NMS_CANDIDATES = 30000;
    private static final int MAXIMUM_DETECTIONS = 300;
    private static final int LETTERBOX_COLOR = 114;
    private static final int XNNPACK_FLAG_QS8 = 0x1;
    private static final int XNNPACK_FLAG_FORCE_FP16 = 0x4;
    private static final float YOLO_11_INPUT_SCALE = 0.00392012158408761f;
    private static final int YOLO_11_INPUT_ZERO_POINT = -128;
    private static final String SIGNATURE = "serving_default";
    private static final String INPUT_NAME = "args_0";
    private static final String OUTPUT_NAME = "output_0";

    private final CompiledModel model;
    private final TensorBuffer inputBuffer;
    private final TensorBuffer outputBuffer;
    private final Profile profile;
    private final List<String> labels;

    LiteRtYoloDetector(Context context, File modelFile, Profile detectorProfile)
            throws Exception {
        if (!modelFile.isFile() || modelFile.length() == 0) {
            throw new IllegalArgumentException("The imported LiteRT model is unavailable.");
        }
        profile = detectorProfile;
        labels = readLabels(context);

        CompiledModel loadedModel = null;
        TensorBuffer loadedInput = null;
        TensorBuffer loadedOutput = null;
        try {
            CompiledModel.Options options = new CompiledModel.Options(Accelerator.CPU);
            options.setCpuOptions(new CompiledModel.CpuOptions(
                    1,
                    profile.xnnPackFlags,
                    null
            ));
            loadedModel = CompiledModel.create(modelFile.getAbsolutePath(), options);
            validateTensorTypes(loadedModel, profile);
            loadedInput = loadedModel.createInputBuffer(INPUT_NAME, SIGNATURE);
            loadedOutput = loadedModel.createOutputBuffer(OUTPUT_NAME, SIGNATURE);
            model = loadedModel;
            inputBuffer = loadedInput;
            outputBuffer = loadedOutput;
        } catch (Exception exception) {
            closeQuietly(loadedOutput);
            closeQuietly(loadedInput);
            closeQuietly(loadedModel);
            throw exception;
        }
    }

    @Override
    public DetectionResult detect(Bitmap source, float confidenceThreshold) throws Exception {
        if (source == null || source.isRecycled()
                || source.getWidth() < 1 || source.getHeight() < 1) {
            throw new IllegalArgumentException("Choose a valid photo before detection.");
        }
        if (confidenceThreshold < 0.0f || confidenceThreshold > 1.0f) {
            throw new IllegalArgumentException("The confidence threshold must be between 0 and 1.");
        }

        PreparedInput prepared = prepareInput(source, profile.inputType);
        if (profile.inputType == InputType.FLOAT32) {
            inputBuffer.writeFloat(prepared.floatData());
        } else {
            inputBuffer.writeInt8(prepared.int8Data());
        }

        long started = System.nanoTime();
        model.run(
                List.of(inputBuffer),
                List.of(outputBuffer),
                SIGNATURE
        );
        long elapsedMilliseconds = Math.round((System.nanoTime() - started) / 1_000_000.0);

        float[] output = outputBuffer.readFloat();
        List<Candidate> candidates = switch (profile.decoder) {
            case YOLO_26 -> decodeYolo26(output, confidenceThreshold, prepared);
            case YOLO_11 -> decodeYolo11(output, confidenceThreshold, prepared);
        };
        List<Detection> detections = nonMaximumSuppression(candidates, profile.iouThreshold);
        return new DetectionResult(elapsedMilliseconds, detections);
    }

    private static void validateTensorTypes(CompiledModel model, Profile profile) {
        TensorType inputType = model.getInputTensorType(INPUT_NAME, SIGNATURE);
        TensorType outputType = model.getOutputTensorType(OUTPUT_NAME, SIGNATURE);
        TensorType.ElementType expectedInput = profile.inputType == InputType.FLOAT32
                ? TensorType.ElementType.FLOAT
                : TensorType.ElementType.INT8;
        if (inputType.getElementType() != expectedInput
                || !inputType.getLayout().getDimensions().equals(List.of(1, 3, 640, 640))) {
            throw new IllegalArgumentException(
                    "Unexpected LiteRT input tensor: " + inputType
            );
        }
        if (outputType.getElementType() != TensorType.ElementType.FLOAT
                || !outputType.getLayout().getDimensions().equals(profile.outputDimensions)) {
            throw new IllegalArgumentException(
                    "Unexpected LiteRT output tensor: " + outputType
            );
        }
    }

    private List<Candidate> decodeYolo26(float[] output, float threshold,
                                         PreparedInput prepared) {
        int expectedValues = YOLO_26_CANDIDATE_COUNT * YOLO_26_FEATURE_COUNT;
        if (output.length != expectedValues) {
            throw new IllegalStateException(
                    "Unexpected YOLO26 output value count: " + output.length
            );
        }
        List<Candidate> decoded = new ArrayList<>();
        for (int candidateIndex = 0;
             candidateIndex < YOLO_26_CANDIDATE_COUNT;
             candidateIndex++) {
            int offset = candidateIndex * YOLO_26_FEATURE_COUNT;
            float score = output[offset + 4];
            int classIndex = Math.round(output[offset + 5]);
            if (score >= threshold && classIndex >= 0 && classIndex < CLASS_COUNT) {
                decoded.add(candidateFromCorners(
                        output[offset],
                        output[offset + 1],
                        output[offset + 2],
                        output[offset + 3],
                        score,
                        classIndex,
                        prepared
                ));
            }
        }
        return decoded;
    }

    private List<Candidate> decodeYolo11(float[] output, float threshold,
                                         PreparedInput prepared) {
        int expectedValues = YOLO_11_CANDIDATE_COUNT * YOLO_11_FEATURE_COUNT;
        if (output.length != expectedValues) {
            throw new IllegalStateException(
                    "Unexpected YOLO11 output value count: " + output.length
            );
        }
        List<Candidate> decoded = new ArrayList<>();
        for (int candidateIndex = 0;
             candidateIndex < YOLO_11_CANDIDATE_COUNT;
             candidateIndex++) {
            float centerX = featureValue(output, candidateIndex, 0);
            float centerY = featureValue(output, candidateIndex, 1);
            float width = featureValue(output, candidateIndex, 2);
            float height = featureValue(output, candidateIndex, 3);
            for (int classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
                float score = probability(featureValue(
                        output,
                        candidateIndex,
                        4 + classIndex
                ));
                if (score >= threshold) {
                    decoded.add(candidateFromCenter(
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
        }
        return decoded;
    }

    private static float featureValue(float[] output, int candidateIndex, int featureIndex) {
        return output[featureIndex * YOLO_11_CANDIDATE_COUNT + candidateIndex];
    }

    private Candidate candidateFromCenter(float centerX, float centerY,
                                          float width, float height,
                                          float score, int classIndex,
                                          PreparedInput prepared) {
        return candidateFromCorners(
                centerX - width / 2.0f,
                centerY - height / 2.0f,
                centerX + width / 2.0f,
                centerY + height / 2.0f,
                score,
                classIndex,
                prepared
        );
    }

    private Candidate candidateFromCorners(float left, float top, float right, float bottom,
                                           float score, int classIndex,
                                           PreparedInput prepared) {
        left = (left - prepared.padX()) / prepared.scaleX();
        right = (right - prepared.padX()) / prepared.scaleX();
        top = (top - prepared.padY()) / prepared.scaleY();
        bottom = (bottom - prepared.padY()) / prepared.scaleY();
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

    private static PreparedInput prepareInput(Bitmap source, InputType inputType) {
        float scale = Math.min(
                INPUT_SIZE / (float) source.getWidth(),
                INPUT_SIZE / (float) source.getHeight()
        );
        int resizedWidth = Math.min(
                (int) Math.ceil(source.getWidth() * scale),
                INPUT_SIZE
        );
        int resizedHeight = Math.min(
                (int) Math.ceil(source.getHeight() * scale),
                INPUT_SIZE
        );
        int padX = (INPUT_SIZE - resizedWidth) / 2;
        int padY = (INPUT_SIZE - resizedHeight) / 2;

        Bitmap resized = Bitmap.createScaledBitmap(
                source,
                resizedWidth,
                resizedHeight,
                true
        );
        Bitmap prepared = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(prepared);
        canvas.drawColor(Color.rgb(LETTERBOX_COLOR, LETTERBOX_COLOR, LETTERBOX_COLOR));
        canvas.drawBitmap(resized, padX, padY, new Paint(Paint.FILTER_BITMAP_FLAG));
        if (resized != source) {
            resized.recycle();
        }

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        prepared.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        prepared.recycle();

        int planeSize = INPUT_SIZE * INPUT_SIZE;
        float[] floatData = inputType == InputType.FLOAT32
                ? new float[planeSize * 3]
                : null;
        byte[] int8Data = inputType == InputType.INT8
                ? new byte[planeSize * 3]
                : null;
        for (int pixelIndex = 0; pixelIndex < planeSize; pixelIndex++) {
            int color = pixels[pixelIndex];
            writeChannel(floatData, int8Data, pixelIndex, Color.red(color));
            writeChannel(floatData, int8Data, planeSize + pixelIndex, Color.green(color));
            writeChannel(floatData, int8Data, 2 * planeSize + pixelIndex, Color.blue(color));
        }
        return new PreparedInput(
                floatData,
                int8Data,
                source.getWidth(),
                source.getHeight(),
                resizedWidth / (float) source.getWidth(),
                resizedHeight / (float) source.getHeight(),
                padX,
                padY
        );
    }

    private static void writeChannel(float[] floatData, byte[] int8Data,
                                     int index, int channelValue) {
        float normalized = channelValue / 255.0f;
        if (floatData != null) {
            floatData[index] = normalized;
        } else {
            int quantized = Math.round(normalized / YOLO_11_INPUT_SCALE)
                    + YOLO_11_INPUT_ZERO_POINT;
            int8Data[index] = (byte) Math.max(Byte.MIN_VALUE, Math.min(Byte.MAX_VALUE, quantized));
        }
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

    private static void closeQuietly(AutoCloseable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (Exception ignored) {
        }
    }

    @Override
    public void close() {
        closeQuietly(outputBuffer);
        closeQuietly(inputBuffer);
        closeQuietly(model);
    }

    enum Profile {
        YOLO_26_FP16(
                InputType.FLOAT32,
                Decoder.YOLO_26,
                0.4f,
                XNNPACK_FLAG_FORCE_FP16,
                List.of(1, 300, 6)
        ),
        YOLO_26_INT8_WEIGHT_ONLY(
                InputType.FLOAT32,
                Decoder.YOLO_26,
                0.4f,
                XNNPACK_FLAG_QS8 | XNNPACK_FLAG_FORCE_FP16,
                List.of(1, 300, 6)
        ),
        YOLO_11_INT8(
                InputType.INT8,
                Decoder.YOLO_11,
                0.7f,
                null,
                List.of(1, 84, 8400)
        );

        private final InputType inputType;
        private final Decoder decoder;
        private final float iouThreshold;
        private final Integer xnnPackFlags;
        private final List<Integer> outputDimensions;

        Profile(InputType inputType, Decoder decoder, float iouThreshold,
                Integer xnnPackFlags, List<Integer> outputDimensions) {
            this.inputType = inputType;
            this.decoder = decoder;
            this.iouThreshold = iouThreshold;
            this.xnnPackFlags = xnnPackFlags;
            this.outputDimensions = outputDimensions;
        }
    }

    private enum InputType {
        FLOAT32,
        INT8
    }

    private enum Decoder {
        YOLO_26,
        YOLO_11
    }

    private record PreparedInput(
            float[] floatData,
            byte[] int8Data,
            int originalWidth,
            int originalHeight,
            float scaleX,
            float scaleY,
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
