package org.arm.learningpath.objectdetection;

record ModelDescriptor(
        String id,
        String displayName,
        String adapterId,
        String runtimeDisplayName,
        String fileName,
        String configurationId,
        int defaultConfidencePercent
) {
    String displayLabel() {
        return displayName + " · " + runtimeDisplayName;
    }
}
