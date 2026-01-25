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
# RTX 4090 OPTIMIZATIONS
# ===============================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/reducto/RolmOCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None

# ===============================
# RTX 4090 PERFORMANCE SETTINGS
# ===============================
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def log(msg):
    print(f"[HANDLER] {msg}", flush=True)


# ===============================
# IMAGE DECODING (4090 OPTIMIZED)
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    
    # RTX 4090 24GB can handle large images
    target_width = 1920
    
    if img.width > target_width:
        scale = target_width / img.width
        new_height = int(img.height * scale)
        img = img.resize(
            (target_width, new_height),
            Image.LANCZOS
        )
    
    return img


# ===============================
# LOAD MODEL (4090 OPTIMIZED)
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

    log("Loading model on RTX 4090...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,  # FP16 for 4090
        local_files_only=True
    )

    model.eval()
    
    # Warmup inference
    log("Warming up RTX 4090...")
    dummy_img = Image.new('RGB', (1920, 1080), color='white')
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
    
    # GPU stats
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}")
        log(f"VRAM Total: {props.total_memory / 1024**3:.1f} GB")
        log(f"VRAM Used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    log("RTX 4090 ready!")


# ===============================
# OCR IMAGE (4090 OPTIMIZED)
# ===============================
@torch.inference_mode()
def ocr_image(image: Image.Image, max_tokens: int = 1500) -> str:
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

    # Use automatic mixed precision
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
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
# HANDLER
# ===============================
def handler(event):
    try:
        load_model()

        input_data = event.get("input", {})
        
        if "image" not in input_data:
            return {
                "status": "error",
                "message": "Missing 'image' field in input"
            }

        max_tokens = input_data.get("max_tokens", 1500)
        
        image = decode_image(input_data["image"])
        text = ocr_image(image, max_tokens=max_tokens)

        gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

        return {
            "status": "success",
            "text": text,
            "gpu_memory_gb": round(gpu_memory, 2)
        }
    
    except Exception as e:
        log(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }


# ===============================
# PRELOAD
# ===============================
log("="*60)
log("RTX 4090 OCR HANDLER - PyTorch 2.4.0")
log("="*60)
load_model()
log("Handler ready for requests!")

runpod.serverless.start({
    "handler": handler
})
