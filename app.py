import streamlit as st
import cv2
import numpy as np
from fpdf import FPDF
import torch
import os
from ultralytics import YOLO

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

def main():
    st.title("Measurement-Tool for Object Detection")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        
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

                object_measurements = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "width": int(w),
                    "height": int(h),
                    "real_width": real_width,
                    "real_height": real_height,
                    "bbox": (int(x1), int(y1), int(w), int(h)),
                    "image": cv2.rectangle(image.copy(), (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
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
        else:
            st.error("No objects detected. Please try again with a different image.")

if __name__ == "__main__":
    main()
