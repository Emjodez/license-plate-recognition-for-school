if __name__ == '__main__':
    import cv2
    import easyocr
    from ultralytics import YOLO as yolo
    import matplotlib.pyplot as plt
    import numpy as np

    model = yolo("runs/detect/train9/weights/best.pt")

    image_path = "octavia-skoda-number-plate-poland.jpg"
    results = model.predict(image_path)

    image = cv2.imread(image_path)

    reader = easyocr.Reader(['en'])

    for result in results:
        # Assuming there's only one result for the license plate
        for box in result.boxes:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # crop license plate from image
            license_plate_region = image[y1:y2, x1:x2]

            plt.imshow(license_plate_region)
            plt.show()
            # Find text using OCR
            ocr_result = reader.readtext(license_plate_region, low_text=0.3, allowlist ='0123456789ZXCVBNMASDFGHJKLQWERTYUIOP')
            detected_text = ''
            # Extract and print the text
            for (bbox, text, prob) in ocr_result:
                temp = ""
                for i in range(len(text)):
                    if text[i] == "S" or text[i] == "5":
                        temp += "S"
                    else:
                        temp += text[i]
                detected_text += temp + ' albo ' + text
                print(f"Detected License Plate Text: {temp}")
            print(detected_text)
            # Annotate the image with the detected text above the license plate
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(detected_text, font, font_scale, font_thickness)[0]
            text_width, text_height = text_size

            # Calculate text position
            text_x = x1 + (x2 - x1) // 2 - text_width // 2
            text_y = y1 - 10  # Place text slightly above the license plate

            # Draw text background rectangle
            cv2.rectangle(image, (text_x - 5, text_y - text_height - 5),
                          (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)

            # Draw text
            cv2.putText(image, detected_text, (text_x, text_y),
                        font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Rejestracja pojazdu')
            plt.axis('off')  # Hide axis
            plt.show()
