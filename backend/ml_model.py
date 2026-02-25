import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64

class ImageGenerator:
    def __init__(self):
        # Load the model (using a smaller, faster model for demonstration)
        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        
        # Optional: Optimize for lower memory usage
        self.pipe.enable_attention_slicing()

    def generate_image_base64(self, prompt: str) -> str:
        # Generate the image
        image = self.pipe(prompt).images[0]
        
        # Convert to Base64 for API transmission
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str