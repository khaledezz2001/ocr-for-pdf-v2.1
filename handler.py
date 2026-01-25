import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ===============================
# OFFLINE MODE (RUNTIME)
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["TRANSFORMERS_CACHE"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/reducto/RolmOCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None

# ===============================
# RTX SPEED / STABILITY
# ===============================
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Added for speed


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


# ===============================
# IMAGE DECODING (OPTIMIZED)
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    
    # Reduced from 1600 to 1280 for faster processing
    # Adjust based on your document quality needs
    target_width = 1280
    
    # Only resize if image is larger
    if img.width > target_width:
        scale = target_width / img.width
        img = img.resize(
            (target_width, int(img.height * scale)),
            Image.LANCZOS  # Changed from BICUBIC for better quality/speed balance
        )
    return img


# ===============================
# LOAD MODEL ONCE
# ===============================
def load_model():
    global processor, model
    if model is not None:
        return

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    log("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=True
    )

    model.eval()
    
    # Warm up the model with a dummy inference
    if DEVICE == "cuda":
        log("Warming up model...")
        dummy_img = Image.new('RGB', (1280, 720), color='white')
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "test"}
            ]
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[dummy_img], return_tensors="pt").to(DEVICE)
        
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=10)
        
        torch.cuda.empty_cache()
    
    log("RolmOCR model loaded and ready")


# ===============================
# OCR IMAGE (OPTIMIZED)
# ===============================
def ocr_image(image: Image.Image, max_tokens: int = 1200) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Return the COMPLETE plain text of this document from top to bottom, "
                        "including headers, tables, footers, bank details, signatures, stamps, "
                        "emails, phone numbers, and all numbers exactly as written."
                    )
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,  # Greedy decoding for speed
            num_beams=1  # Explicit single beam
        )

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    # Clean output
    if "assistant" in decoded:
        decoded = decoded.split("assistant", 1)[-1]

    decoded = decoded.strip()
    while decoded.startswith(("system\n", "user\n", ".")):
        decoded = decoded.split("\n", 1)[-1].strip()

    decoded = decoded.replace("assistant\n", "", 1).strip()

    return decoded


# ===============================
# HANDLER (OPTIMIZED)
# ===============================
def handler(event):
    try:
        load_model()

        input_data = event.get("input", {})
        
        if "image" not in input_data:
            return {
                "status": "error",
                "message": "Only image input is supported"
            }

        # Optional: allow custom max_tokens
        max_tokens = input_data.get("max_tokens", 1200)
        
        image = decode_image(input_data["image"])
        text = ocr_image(image, max_tokens=max_tokens)

        return {
            "status": "success",
            "text": text
        }
    
    except Exception as e:
        log(f"Error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


# ===============================
# PRELOAD MODEL AT STARTUP
# ===============================
log("Preloading model at startup...")
load_model()
log("Handler ready to accept requests")

# Start RunPod serverless
runpod.serverless.start({
    "handler": handler
})
