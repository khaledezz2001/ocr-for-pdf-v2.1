import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ===============================
# OFFLINE MODE (RUNTIME )
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


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


# ===============================
# IMAGE DECODING
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    target_width = 1600
    scale = target_width / img.width
    img = img.resize(
        (target_width, int(img.height * scale)),
        Image.BICUBIC
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
    log("RolmOCR model loaded")


# ===============================
# OCR IMAGE
# ===============================
def ocr_image(image: Image.Image) -> str:
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
            max_new_tokens=1200,
            temperature=0.1
        )

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    if "assistant" in decoded:
        decoded = decoded.split("assistant", 1)[-1]

    decoded = decoded.strip()
    while decoded.startswith(("system\n", "user\n", ".")):
        decoded = decoded.split("\n", 1)[-1].strip()

    # CLEAN FINAL OUTPUT
    decoded = decoded.replace("assistant\n", "", 1).strip()

    return decoded


# ===============================
# HANDLER (IMAGE ONLY → TEXT ONLY)
# ===============================
def handler(event):
    load_model()

    if "image" not in event["input"]:
        return {
            "status": "error",
            "message": "Only image input is supported"
        }

    image = decode_image(event["input"]["image"])
    text = ocr_image(image)

    return {
        "status": "success",
        "text": text
    }


# ===============================
# PRELOAD
# ===============================
log("Preloading model...")
load_model()

runpod.serverless.start({
    "handler": handler
})
