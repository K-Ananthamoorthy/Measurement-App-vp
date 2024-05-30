import streamlit as st
import cv2
import numpy as np
from fpdf import FPDF
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load YOLOv8 model
def load_yolo_model():
    model = YOLO('yolov8n.pt')  # Using the small version of YOLOv8
    return model

# Perform object detection
def detect_objects(image, model):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb)  # Correct usage for YOLOv8

    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    class_names = model.names  # Access class names from the model

    return boxes, confidences, class_ids, class_names

def save_as_pdf(output_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Object Detection Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, f"Object: {output_dict['class_name']}", ln=True)
    pdf.cell(200, 10, f"Confidence: {output_dict['confidence']:.2f}", ln=True)
    pdf.cell(200, 10, f"Width: {output_dict['width']} pixels", ln=True)
    pdf.cell(200, 10, f"Height: {output_dict['height']} pixels", ln=True)
    pdf.cell(200, 10, f"Real Width: {output_dict['real_width']:.2f} cm", ln=True)
    pdf.cell(200, 10, f"Real Height: {output_dict['real_height']:.2f} cm", ln=True)
    pdf.ln(10)

    x, y, w, h = output_dict['bbox']
    pdf.cell(200, 10, f"Bounding Box (x, y, width, height): ({x}, {y}, {w}, {h})", ln=True)
    
    # Save the bounding box image
    bbox_image = output_dict['image']
    cv2.imwrite("detected_object.jpg", bbox_image)

    # Add the image with bounding box to the PDF
    pdf.image("detected_object.jpg", x=10, y=80, w=150)
    
    pdf.output("object_detection_report.pdf")

def draw_grid(image, grid_size):
    h, w = image.shape[:2]
    for i in range(0, w, grid_size):
        cv2.line(image, (i, 0), (i, h), (0, 255, 0), 1)
    for i in range(0, h, grid_size):
        cv2.line(image, (0, i), (w, i), (0, 255, 0), 1)
    return image

def main():
    st.title("Object Detection and Measurement Tool")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

        # Get grid size from user
        grid_size = st.slider("Select Grid Size", 10, 100, 20)

        # Draw grid on the image
        image_with_grid = draw_grid(image.copy(), grid_size)
        st.image(image_with_grid, channels="BGR", caption="Image with Grid")

        # Convert to PIL Image for st_canvas background
        pil_image = Image.fromarray(cv2.cvtColor(image_with_grid, cv2.COLOR_BGR2RGB))

        model = load_yolo_model()
        boxes, confidences, class_ids, class_names = detect_objects(image, model)

        if len(boxes) > 0:
            # Assuming a reference object is included in the image for real-world measurement
            # For simplicity, let's assume 1 pixel = 0.026 cm (for a specific known reference)
            pixel_to_cm_ratio = 0.026  # This should be calculated based on a known reference in the image

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i][:4]
                w = x2 - x1
                h = y2 - y1
                class_name = class_names[class_ids[i]]
                confidence = confidences[i]
                
                real_width = w * pixel_to_cm_ratio
                real_height = h * pixel_to_cm_ratio

                # Add measurement details to the image
                output_image = image.copy()
                cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                text = f"{class_name}: {confidence:.2f}\nW: {w}px H: {h}px\nRW: {real_width:.2f}cm RH: {real_height:.2f}cm"
                y_offset = y1 - 10 if y1 - 10 > 10 else y1 + 10
                for i, line in enumerate(text.split('\n')):
                    cv2.putText(output_image, line, (int(x1), int(y_offset + i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                object_measurements = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "width": int(w),
                    "height": int(h),
                    "real_width": real_width,
                    "real_height": real_height,
                    "bbox": (int(x1), int(y1), int(w), int(h)),
                    "image": output_image
                }

                # Display measurements on Streamlit webpage
                st.image(object_measurements['image'], channels="BGR")
                st.write(f"Detected Object: {class_name}")
                st.write(f"Confidence: {confidence:.2f}")
                st.write(f"Width: {int(w)} pixels")
                st.write(f"Height: {int(h)} pixels")
                st.write(f"Real Width: {real_width:.2f} cm")
                st.write(f"Real Height: {real_height:.2f} cm")

                save_as_pdf(object_measurements)
                st.download_button("Download PDF Report", data=open("object_detection_report.pdf", "rb").read(), file_name="object_detection_report.pdf")
                break

        # Selecting grids with drawable canvas
        st.subheader("Select Grid Area for Measurement")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill
            stroke_width=1,
            stroke_color="#FF0000",
            background_image=pil_image,
            update_streamlit=True,
            height=image.shape[0],
            width=image.shape[1],
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[0]
                if obj["width"] > 0 and obj["height"] > 0:
                    left = int(obj["left"])
                    top = int(obj["top"])
                    width = int(obj["width"])
                    height = int(obj["height"])

                    x1, y1, x2, y2 = left, top, left + width, top + height

                    # Ensure the selected area is within image bounds
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 > image.shape[1]: x2 = image.shape[1]
                    if y2 > image.shape[0]: y2 = image.shape[0]

                    if x1 < x2 and y1 < y2:
                        selected_area = image[y1:y2, x1:x2]
                        st.image(selected_area, channels="BGR", caption="Selected Grid Area")

                        # Calculate the measurements of the selected area
                        w_selected = x2 - x1
                        h_selected = y2 - y1
                        real_width_selected = w_selected * pixel_to_cm_ratio
                        real_height_selected = h_selected * pixel_to_cm_ratio

                        # Create output image for selected area
                        output_image_selected = image.copy()
                        cv2.rectangle(output_image_selected, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        text_selected = f"Selected Area\nW: {w_selected}px H: {h_selected}px\nRW: {real_width_selected:.2f}cm RH: {real_height_selected:.2f}cm"
                        y_offset_selected = y1 - 10 if y1 - 10 > 10 else y1 + 10
                        for i, line in enumerate(text_selected.split('\n')):
                            cv2.putText(output_image_selected, line, (x1, y_offset_selected + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        st.image(output_image_selected, channels="BGR", caption="Image with Selected Area")
                        st.write(f"Selected Area Measurements:")
                        st.write(f"Width: {w_selected} pixels")
                        st.write(f"Height: {h_selected} pixels")
                        st.write(f"Real Width: {real_width_selected:.2f} cm")
                        st.write(f"Real Height: {real_height_selected:.2f} cm")

if __name__ == "__main__":
    main()
