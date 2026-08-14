package org.arm.learningpath.objectdetection;

import java.util.List;
import java.util.Locale;

record DetectionResult(long elapsedMilliseconds, List<Detection> detections) {
    DetectionResult {
        detections = List.copyOf(detections);
    }

    String summary() {
        StringBuilder text = new StringBuilder();
        text.append("Processing time: ").append(elapsedMilliseconds).append(" ms\n");
        text.append("Detections: ").append(detections.size()).append("\n\n");
        int count = Math.min(10, detections.size());
        for (int index = 0; index < count; index++) {
            Detection detection = detections.get(index);
            text.append(String.format(
                    Locale.US,
                    "%d. %5.1f%%  %s%n",
                    index + 1,
                    detection.score() * 100.0f,
                    detection.label()
            ));
        }
        if (detections.size() > count) {
            text.append("…and ").append(detections.size() - count).append(" more\n");
        }
        return text.toString();
    }
}
