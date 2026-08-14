package org.arm.learningpath.objectdetection;

record Detection(
        float left,
        float top,
        float right,
        float bottom,
        float score,
        int classIndex,
        String label
) {
}
