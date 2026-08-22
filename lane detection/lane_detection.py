# -*- coding: utf-8 -*-

"""
Lane Line Detection

Pipeline:
1. Grayscale
2. Gaussian Blur
3. Canny Edge Detection
4. Region of Interest
5. Hough Line Transform
6. Average Left/Right Lines
7. Draw Lane Lines
8. Save Output Video
"""

import cv2
import numpy as np


# ============================================================
# 1. GRAYSCALE
# ============================================================

def grayscale(img):
    """
    Convert BGR image into grayscale.
    """

    return cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


# ============================================================
# 2. CANNY EDGE DETECTION
# ============================================================

def canny(
    img,
    low_threshold=50,
    high_threshold=150
):
    """
    Detect edges using Canny algorithm.
    """

    return cv2.Canny(
        img,
        low_threshold,
        high_threshold
    )


# ============================================================
# 3. GAUSSIAN BLUR
# ============================================================

def gaussian_blur(
    img,
    kernel_size=5
):
    """
    Remove noise using Gaussian Blur.
    """

    return cv2.GaussianBlur(
        img,
        (
            kernel_size,
            kernel_size
        ),
        0
    )


# ============================================================
# 4. REGION OF INTEREST
# ============================================================

def region_of_interest(img):
    """
    Keep only the road area.
    """

    height = img.shape[0]
    width = img.shape[1]

    polygons = np.array([
        [
            (
                int(0.10 * width),
                height
            ),

            (
                int(0.45 * width),
                int(0.60 * height)
            ),

            (
                int(0.55 * width),
                int(0.60 * height)
            ),

            (
                int(0.90 * width),
                height
            )
        ]
    ], dtype=np.int32)

    # Create black mask
    mask = np.zeros_like(img)

    # Fill road region
    cv2.fillPoly(
        mask,
        polygons,
        255
    )

    # Apply mask
    masked_image = cv2.bitwise_and(
        img,
        mask
    )

    return masked_image


# ============================================================
# 5. DISPLAY / DRAW LINES
# ============================================================

def display_lines(
    img,
    lines,
    color=(255, 0, 0),
    thickness=10
):
    """
    Draw detected lane lines.
    """

    line_image = np.zeros_like(img)

    # No lines detected
    if lines is None:
        return line_image

    for line in lines:

        # ----------------------------------------------------
        # IMPORTANT FIX
        #
        # HoughLinesP can return:
        #
        # [[x1, y1, x2, y2]]
        #
        # or
        #
        # [x1, y1, x2, y2]
        #
        # reshape(-1) handles both.
        # ----------------------------------------------------

        coordinates = np.array(
            line
        ).reshape(-1)

        # Make sure exactly 4 values exist
        if len(coordinates) != 4:
            continue

        x1, y1, x2, y2 = coordinates

        # Convert NumPy integers to normal Python integers
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Draw line
        cv2.line(
            line_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

    return line_image


# ============================================================
# 6. MAKE COORDINATES
# ============================================================

def make_coordinates(
    img,
    line_parameters
):
    """
    Convert slope/intercept into line coordinates.
    """

    slope, intercept = line_parameters

    # Avoid division by zero
    if abs(slope) < 1e-6:
        return None

    # Bottom of image
    y1 = img.shape[0]

    # Upper point of lane
    y2 = int(
        y1 * 0.60
    )

    # x = (y - b) / m
    x1 = int(
        (y1 - intercept) / slope
    )

    x2 = int(
        (y2 - intercept) / slope
    )

    return np.array(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dtype=np.int32
    )


# ============================================================
# 7. AVERAGE SLOPE AND INTERCEPT
# ============================================================

def average_slope_intercept(
    img,
    lines
):
    """
    Separate left and right lane lines
    and calculate their average.
    """

    left_fit = []
    right_fit = []

    # --------------------------------------------------------
    # IMPORTANT:
    # HoughLinesP can return None.
    # --------------------------------------------------------

    if lines is None:
        return []

    # --------------------------------------------------------
    # Loop through detected lines
    # --------------------------------------------------------

    for line in lines:

        # ----------------------------------------------------
        # FIX FOR:
        #
        # TypeError:
        # cannot unpack non-iterable numpy.int32 object
        #
        # Flatten the line first.
        # ----------------------------------------------------

        coordinates = np.array(
            line
        ).reshape(-1)

        # We need exactly:
        # x1, y1, x2, y2

        if len(coordinates) != 4:
            continue

        x1, y1, x2, y2 = coordinates

        # Convert to float
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        # ----------------------------------------------------
        # Avoid vertical lines
        # because x2 - x1 = 0
        # ----------------------------------------------------

        if abs(x2 - x1) < 1e-6:
            continue

        # ----------------------------------------------------
        # Calculate slope and intercept
        # ----------------------------------------------------

        slope = (
            (y2 - y1)
            /
            (x2 - x1)
        )

        intercept = (
            y1
            -
            slope * x1
        )

        # ----------------------------------------------------
        # Ignore almost horizontal lines
        # ----------------------------------------------------

        if abs(slope) < 0.5:
            continue

        # ----------------------------------------------------
        # Left lane
        # ----------------------------------------------------

        if slope < 0:

            left_fit.append(
                (
                    slope,
                    intercept
                )
            )

        # ----------------------------------------------------
        # Right lane
        # ----------------------------------------------------

        else:

            right_fit.append(
                (
                    slope,
                    intercept
                )
            )

    # ========================================================
    # Average the lines
    # ========================================================

    averaged_lines = []

    # --------------------------------------------------------
    # Left lane
    # --------------------------------------------------------

    if len(left_fit) > 0:

        left_average = np.average(
            left_fit,
            axis=0
        )

        left_coordinates = make_coordinates(
            img,
            left_average
        )

        if left_coordinates is not None:

            averaged_lines.append(
                left_coordinates
            )

    # --------------------------------------------------------
    # Right lane
    # --------------------------------------------------------

    if len(right_fit) > 0:

        right_average = np.average(
            right_fit,
            axis=0
        )

        right_coordinates = make_coordinates(
            img,
            right_average
        )

        if right_coordinates is not None:

            averaged_lines.append(
                right_coordinates
            )

    return averaged_lines


# ============================================================
# 8. VIDEO PROCESSING
# ============================================================

def process_video(
    input_video,
    output_video
):
    """
    Read input video, detect lanes,
    and save processed video.
    """

    # --------------------------------------------------------
    # Open input video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        input_video
    )

    if not cap.isOpened():

        print(
            f"ERROR: Cannot open video: {input_video}"
        )

        return

    # --------------------------------------------------------
    # Get video properties
    # --------------------------------------------------------

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    # Some videos may report FPS as 0
    if fps <= 0:

        fps = 20.0

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    # --------------------------------------------------------
    # Process video frame by frame
    # --------------------------------------------------------

    frame_count = 0

    while True:

        ret, frame = cap.read()

        # End of video
        if not ret:
            break

        frame_count += 1

        # ----------------------------------------------------
        # 1. Grayscale
        # ----------------------------------------------------

        gray = grayscale(
            frame
        )

        # ----------------------------------------------------
        # 2. Gaussian Blur
        # ----------------------------------------------------

        blur = gaussian_blur(
            gray,
            kernel_size=5
        )

        # ----------------------------------------------------
        # 3. Canny
        # ----------------------------------------------------

        edges = canny(
            blur,
            low_threshold=50,
            high_threshold=150
        )

        # ----------------------------------------------------
        # 4. Region of Interest
        # ----------------------------------------------------

        cropped = region_of_interest(
            edges
        )

        # ----------------------------------------------------
        # 5. Hough Line Transform
        # ----------------------------------------------------

        lines = cv2.HoughLinesP(
            cropped,
            rho=2,
            theta=np.pi / 180,
            threshold=100,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=5
        )

        # ----------------------------------------------------
        # 6. Average lane lines
        # ----------------------------------------------------

        averaged_lines = average_slope_intercept(
            frame,
            lines
        )

        # ----------------------------------------------------
        # 7. Draw lines
        # ----------------------------------------------------

        line_img = display_lines(
            frame,
            averaged_lines,
            color=(255, 0, 0),
            thickness=10
        )

        # ----------------------------------------------------
        # 8. Combine original + detected lanes
        # ----------------------------------------------------

        combo = cv2.addWeighted(
            frame,
            0.8,
            line_img,
            1.0,
            1.0
        )

        # ----------------------------------------------------
        # 9. Save frame
        # ----------------------------------------------------

        out.write(
            combo
        )

        # ----------------------------------------------------
        # Display progress
        # ----------------------------------------------------

        if frame_count % 50 == 0:

            print(
                f"Processed {frame_count} frames..."
            )

    # --------------------------------------------------------
    # Release resources
    # --------------------------------------------------------

    cap.release()

    out.release()

    cv2.destroyAllWindows()

    print()
    print(
        "=========================================="
    )

    print(
        "Processing complete!"
    )

    print(
        f"Output saved as: {output_video}"
    )

    print(
        f"Total frames processed: {frame_count}"
    )

    print(
        "=========================================="
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    input_video = "solidWhiteRight.mp4"

    output_video = "lane_detected_output.mp4"

    process_video(
        input_video,
        output_video
    )