import base64
import io
import os
import json
import cv2
import requests
import numpy as np
from PIL import Image, ImageDraw
from google import genai
from src.FUNCTION.Tools.get_env import EnvManager
import re 


class ImageProcessor:
    def __init__(self, image_path="captured_image.png", model_name="gemini-2.5-flash"):
        self.image_path = image_path
        self.require_width = 336
        self.require_height = 336
        self.model = model_name  # Set model name here
        self.client = genai.Client(api_key=EnvManager.load_variable("genai_key"))

    # ---------- Image capture and resizing ----------
    def resize_image(self, require_width=None, require_height=None) -> bool:
        require_width = require_width or self.require_width
        require_height = require_height or self.require_height

        try:
            with Image.open(self.image_path) as img:
                width, height = img.size
                if height <= require_height and width <= require_width:
                    return True
                img = img.resize((require_width, require_height), Image.ANTIALIAS)
                img.save(self.image_path)
                print(f"Image saved to {self.image_path}, size: {require_width}x{require_height}")
        except Exception as e:
            print(e)
            return False
        return True

    def capture_image_and_save(self) -> str | None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None
        try:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(self.image_path, frame)
                print(f"Image captured and saved as {self.image_path}")
                return self.image_path
            else:
                print("Error: Could not capture image.")
                return None
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ---------- Basic detection ----------
    def detect_image(self , query:str) -> str | None:
        """Detect content using the set model."""
        if not query:
            query = "What is this image?"
        try:
            image = Image.open(self.image_path)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[query, image]
            )
            return response.text
        except Exception as e:
            print(f"Error: {e}")
            return None

    # ---------- Object detection ----------
    def detect_objects(self, prompt="Detect all prominent items in the image.") -> list | None:
        try:
            image = Image.open(self.image_path)
            config = genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image],
                config=config
            )
            width, height = image.size
            boxes = json.loads(response.text)
            converted_boxes = []
            for box in boxes:
                y1, x1, y2, x2 = box["box_2d"]
                converted_boxes.append([
                    int(x1/1000*width), int(y1/1000*height),
                    int(x2/1000*width), int(y2/1000*height)
                ])
            return converted_boxes
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def extract_segmentation_masks(self, prompt=None, output_dir="segmentation_results"):
        try:
            if not self.image_path and not self.image_obj:
                print("No image provided.")
                return None

            # Load image
            image = Image.open(self.image_path or self.image_obj)
            image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

            # Default prompt
            prompt = prompt or "Give segmentation masks for prominent items."
            # Enforce structured JSON output
            prompt = f"""{prompt}
            Output ONLY a JSON list of segmentation masks.
            Each entry should have:
            - "box_2d": [y0, x0, y1, x1]
            - "mask": base64 PNG string
            - "label": descriptive text
            Do not include any extra text or explanations."""

            # Prepare model config
            config = genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig()
            )

            # Call the model
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image],
                config=config
            )

            # Parse JSON safely
            try:
                items = json.loads(self._parse_json(response.text))
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                items = []

            os.makedirs(output_dir, exist_ok=True)
            composite_images = []  # Store overlay images for Streamlit

            for i, item in enumerate(items):
                if not all(k in item for k in ("box_2d", "mask", "label")):
                    print(f"Skipping item {i}: missing required keys")
                    continue

                y0, x0, y1, x1 = item.get("box_2d", [0, 0, image.height, image.width])
                y0 = int(y0 / 1000 * image.size[1])
                x0 = int(x0 / 1000 * image.size[0])
                y1 = int(y1 / 1000 * image.size[1])
                x1 = int(x1 / 1000 * image.size[0])
                if y0 >= y1 or x0 >= x1:
                    continue

                # Decode mask
                png_str = item["mask"]
                if not png_str.startswith("data:image/png;base64,"):
                    continue
                png_str = png_str.removeprefix("data:image/png;base64,")
                mask_data = base64.b64decode(png_str)
                mask = Image.open(io.BytesIO(mask_data)).resize((x1-x0, y1-y0), Image.Resampling.BILINEAR)

                # Create overlay
                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                mask_array = np.array(mask)
                color = (255, 255, 255, 200)
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        if mask_array[y - y0, x - x0] > 128:
                            overlay_draw.point((x, y), fill=color)

                # Save mask and overlay
                mask_filename = f"{item['label']}_{i}_mask.png"
                overlay_filename = f"{item['label']}_{i}_overlay.png"
                mask.save(os.path.join(output_dir, mask_filename))
                composite = Image.alpha_composite(image.convert('RGBA'), overlay)
                composite.save(os.path.join(output_dir, overlay_filename))

                composite_images.append(composite)  # For Streamlit
                print(f"Saved mask and overlay for {item['label']} to {output_dir}")

            # Convert first composite image to base64 for Streamlit display
            if composite_images:
                buffered = io.BytesIO()
                composite_images[0].save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                return img_base64

            return None

        except Exception as e:
            print(f"Error in extract_segmentation_masks: {e}")
            return None
        
    # ---------- Helper ----------
    def _parse_json(self, json_output: str):
        """
        Extracts the first JSON array or object from a model output,
        even if there is extra text around it.
        """


        # Remove markdown code fences if present
        json_output = re.sub(r"```(json)?", "", json_output, flags=re.IGNORECASE).strip()

        # Find the first JSON object or array
        start = json_output.find("[")
        end = json_output.rfind("]") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(json_output[start:end])
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                return []
        else:
            # Try object instead of list
            start = json_output.find("{")
            end = json_output.rfind("}") + 1
            if start != -1 and end != -1:
                try:
                    return [json.loads(json_output[start:end])]
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {e}")
                    return []
        return []


    # ---------- File API ----------
    def upload_image_file(self) -> genai.types.File:
        return self.client.files.upload(file=self.image_path)

    # ---------- Multi-image prompt ----------
    def multi_image_prompt(self, images: list, prompt: str):
        contents = [prompt]
        for img in images:
            if isinstance(img, str):
                contents.append(self.client.files.upload(file=img))
            elif isinstance(img, Image.Image):
                contents.append(img)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents
        )
        return response.text

    # ---------- Inline image from URL ----------
    def get_image_from_url(self, url: str) -> Image.Image:
        img_bytes = requests.get(url).content
        return Image.open(io.BytesIO(img_bytes))


def main():
    # Initialize processor
    processor = ImageProcessor(image_path="test_image.png")

    # ---------- 1. Capture image ----------
    captured_path = processor.capture_image_and_save()
    if captured_path:
        print(f"Captured image saved at: {captured_path}")

    # ---------- 2. Resize image ----------
    if processor.resize_image(336, 336):
        print(f"Image resized successfully.")

    # ---------- 3. Basic detection ----------
    detection = processor.detect_image()
    if detection:
        print("Basic detection output:")
        print(detection)

    # ---------- 4. Object detection ----------
    boxes = processor.detect_objects(prompt="Detect prominent objects and their bounding boxes.")
    if boxes:
        print("Detected bounding boxes:")
        for b in boxes:
            print(b)

    # ---------- 5. Segmentation ----------
    processor.extract_segmentation_masks(
        prompt="Segment all wooden and glass items in the image.",
        output_dir="segmentation_results"
    )

    # ---------- 6. File API upload ----------
    uploaded_file = processor.upload_image_file()
    print(f"Uploaded file ID: {uploaded_file.id}")

    # ---------- 7. Multi-image prompt ----------
    # For demonstration, using the same image twice
    multi_response = processor.multi_image_prompt(
        images=[processor.image_path, processor.image_path],
        prompt="Compare these two images and describe the differences."
    )
    print("Multi-image prompt response:")
    print(multi_response)

    # ---------- 8. Fetch image from URL ----------
    url = "https://via.placeholder.com/150"  # Example image URL
    image_from_url = processor.get_image_from_url(url)
    image_from_url.show()
    print("Fetched and displayed image from URL.")

if __name__ == "__main__":
    main()


