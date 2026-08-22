import cv2
import numpy as np
import tempfile
import streamlit as st


st.set_page_config(
    page_title="Road Lane Detection",
    layout="wide"
)

st.title("🚗 Road Lane Detection (OpenCV)")

st.write(
    "Upload a road video and the app will detect lanes."
)


uploaded_file = st.file_uploader(
    "Upload a road video",
    type=["mp4", "avi", "mov", "mpeg"]
)


def process_video(input_path, output_path):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(
            "Could not open the uploaded video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------------------
        # 1. Grayscale
        # -----------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # -----------------------------------------
        # 2. Gaussian Blur
        # -----------------------------------------

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # -----------------------------------------
        # 3. Canny Edge Detection
        # -----------------------------------------

        edges = cv2.Canny(
            blur,
            50,
            150
        )

        # -----------------------------------------
        # 4. Region of Interest
        # -----------------------------------------

        mask = np.zeros_like(edges)

        polygon = np.array([
            [
                (int(0.10 * width), height),
                (int(0.45 * width), int(0.60 * height)),
                (int(0.55 * width), int(0.60 * height)),
                (int(0.90 * width), height)
            ]
        ], dtype=np.int32)

        cv2.fillPoly(
            mask,
            polygon,
            255
        )

        cropped = cv2.bitwise_and(
            edges,
            mask
        )

        # -----------------------------------------
        # 5. Hough Line Detection
        # -----------------------------------------

        lines = cv2.HoughLinesP(
            cropped,
            rho=2,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=40,
            maxLineGap=5
        )

        # -----------------------------------------
        # 6. Create blank image for lane lines
        # -----------------------------------------

        line_image = np.zeros_like(frame)

        left_lines = []
        right_lines = []

        # -----------------------------------------
        # 7. Process detected lines
        # -----------------------------------------

        if lines is not None:

            for line in lines:

                # IMPORTANT FIX
                #
                # Convert whatever shape OpenCV
                # returns into [x1,y1,x2,y2]

                coordinates = np.asarray(
                    line
                ).reshape(-1)

                if coordinates.size != 4:
                    continue

                x1, y1, x2, y2 = coordinates

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # Avoid vertical lines
                if x2 == x1:
                    continue

                slope = (
                    (y2 - y1)
                    /
                    (x2 - x1)
                )

                intercept = (
                    y1 - slope * x1
                )

                # Ignore horizontal lines
                if abs(slope) < 0.5:
                    continue

                # Left lane
                if slope < 0:

                    left_lines.append(
                        (slope, intercept)
                    )

                # Right lane
                else:

                    right_lines.append(
                        (slope, intercept)
                    )

        # -----------------------------------------
        # 8. Draw averaged left lane
        # -----------------------------------------

        if len(left_lines) > 0:

            slope, intercept = np.mean(
                left_lines,
                axis=0
            )

            if abs(slope) > 1e-6:

                y1 = height
                y2 = int(
                    height * 0.60
                )

                x1 = int(
                    (y1 - intercept) / slope
                )

                x2 = int(
                    (y2 - intercept) / slope
                )

                cv2.line(
                    line_image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    10
                )

        # -----------------------------------------
        # 9. Draw averaged right lane
        # -----------------------------------------

        if len(right_lines) > 0:

            slope, intercept = np.mean(
                right_lines,
                axis=0
            )

            if abs(slope) > 1e-6:

                y1 = height
                y2 = int(
                    height * 0.60
                )

                x1 = int(
                    (y1 - intercept) / slope
                )

                x2 = int(
                    (y2 - intercept) / slope
                )

                cv2.line(
                    line_image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    10
                )

        # -----------------------------------------
        # 10. Combine original and lane lines
        # -----------------------------------------

        result = cv2.addWeighted(
            frame,
            0.8,
            line_image,
            1.0,
            1.0
        )

        # -----------------------------------------
        # 11. Write output frame
        # -----------------------------------------

        out.write(result)

    cap.release()
    out.release()


# ==========================================================
# STREAMLIT UPLOAD
# ==========================================================

if uploaded_file is not None:

    # Save uploaded video temporarily

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as temp_input:

        temp_input.write(
            uploaded_file.read()
        )

        input_path = temp_input.name

    # Output temporary file

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    # Process video

    with st.spinner(
        "Detecting lanes..."
    ):

        try:

            process_video(
                input_path,
                output_path
            )

            st.success(
                "Lane detection completed!"
            )

            # -------------------------------------
            # Show original video
            # -------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Original Video"
                )

                st.video(
                    uploaded_file
                )

            # -------------------------------------
            # Show processed video
            # -------------------------------------

            with col2:

                st.subheader(
                    "Detected Lane Video"
                )

                with open(
                    output_path,
                    "rb"
                ) as video_file:

                    video_bytes = (
                        video_file.read()
                    )

                st.video(
                    video_bytes
                )

                # Download button

                st.download_button(
                    label="⬇️ Download Output Video",
                    data=video_bytes,
                    file_name="lane_detected_output.mp4",
                    mime="video/mp4"
                )

        except Exception as e:

            st.error(
                f"Error while processing video: {e}"
            )