package org.arm.learningpath.objectdetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.List;
import java.util.Locale;

public final class DetectionOverlayView extends View {
    private static final int[] BOX_COLORS = {
            Color.rgb(0, 155, 119),
            Color.rgb(230, 98, 0),
            Color.rgb(0, 114, 178),
            Color.rgb(204, 121, 167),
            Color.rgb(240, 228, 66)
    };

    private final Paint imagePaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelBackgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private Bitmap bitmap;
    private int sourceWidth;
    private int sourceHeight;
    private List<Detection> detections = List.of();

    public DetectionOverlayView(Context context, AttributeSet attributes) {
        super(context, attributes);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(dp(3));
        labelPaint.setColor(Color.WHITE);
        labelPaint.setTextSize(dp(13));
        labelPaint.setFakeBoldText(true);
    }

    void setBitmap(Bitmap selectedBitmap) {
        bitmap = selectedBitmap;
        sourceWidth = selectedBitmap == null ? 0 : selectedBitmap.getWidth();
        sourceHeight = selectedBitmap == null ? 0 : selectedBitmap.getHeight();
        detections = List.of();
        invalidate();
    }

    void setCameraFrameSize(int width, int height) {
        bitmap = null;
        sourceWidth = width;
        sourceHeight = height;
        invalidate();
    }

    void clear() {
        bitmap = null;
        sourceWidth = 0;
        sourceHeight = 0;
        detections = List.of();
        invalidate();
    }

    void setDetections(List<Detection> values) {
        detections = values == null ? List.of() : List.copyOf(values);
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (sourceWidth <= 0 || sourceHeight <= 0) {
            return;
        }

        float scale = Math.min(
                getWidth() / (float) sourceWidth,
                getHeight() / (float) sourceHeight
        );
        float drawnWidth = sourceWidth * scale;
        float drawnHeight = sourceHeight * scale;
        float offsetX = (getWidth() - drawnWidth) / 2.0f;
        float offsetY = (getHeight() - drawnHeight) / 2.0f;
        RectF destination = new RectF(
                offsetX,
                offsetY,
                offsetX + drawnWidth,
                offsetY + drawnHeight
        );
        if (bitmap != null && !bitmap.isRecycled()) {
            canvas.drawBitmap(bitmap, null, destination, imagePaint);
        }

        for (Detection detection : detections) {
            int color = BOX_COLORS[Math.floorMod(detection.classIndex(), BOX_COLORS.length)];
            boxPaint.setColor(color);
            labelBackgroundPaint.setColor(color);
            RectF box = new RectF(
                    offsetX + detection.left() * scale,
                    offsetY + detection.top() * scale,
                    offsetX + detection.right() * scale,
                    offsetY + detection.bottom() * scale
            );
            canvas.drawRect(box, boxPaint);

            String label = String.format(
                    Locale.US,
                    "%s %.0f%%",
                    detection.label(),
                    detection.score() * 100.0f
            );
            float padding = dp(5);
            float textWidth = labelPaint.measureText(label);
            Paint.FontMetrics metrics = labelPaint.getFontMetrics();
            float labelHeight = metrics.descent - metrics.ascent + padding * 2;
            float labelTop = Math.max(offsetY, box.top - labelHeight);
            RectF labelBackground = new RectF(
                    box.left,
                    labelTop,
                    Math.min(getWidth(), box.left + textWidth + padding * 2),
                    labelTop + labelHeight
            );
            canvas.drawRect(labelBackground, labelBackgroundPaint);
            canvas.drawText(
                    label,
                    labelBackground.left + padding,
                    labelBackground.bottom - padding - metrics.descent,
                    labelPaint
            );
        }
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }
}
