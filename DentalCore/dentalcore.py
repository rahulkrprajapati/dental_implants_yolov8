from PIL import Image
from ultralytics import YOLO
import io

class DentalCore:
    def __init__(self) -> None:
        self.model = YOLO('./train/weights/best.pt')

    def resize_image(self, image_bytes: bytes, size: tuple = (416, 416)) -> Image:
        """Resize an image to the specified size."""
        image = Image.open(io.BytesIO(image_bytes))
        return image.resize(size, Image.LANCZOS)
    
    def inference_xray(self, image_bytes):
        results = self.model(image_bytes)
        output_image = None
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            output_image = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        
        return output_image