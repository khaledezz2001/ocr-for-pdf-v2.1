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
# RTX 5090 OPTIMIZATIONS
# ===============================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/reducto/RolmOCR"
DEVICE = "cuda"

processor = None
model = None

# ===============================
# RTX 5090 PERFORMANCE SETTINGS
# ===============================
if torch.cuda.is_available():
    # TF32 for massive speedup on 5090
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Use Flash Attention 2 if available
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Set CUDA device properties for optimal performance
    torch.cuda.set_per_process_memory_fraction(0.95, 0)


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


# ===============================
# IMAGE DECODING (5090 OPTIMIZED)
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    
    # RTX 5090 can handle larger images easily
    # Increased to 2048 for better OCR accuracy
    target_width = 2048
    
    if img.width > target_width:
        scale = target_width / img.width
        new_height = int(img.height * scale)
        img = img.resize(
            (target_width, new_height),
            Image.LANCZOS
        )
    
    return img


# ===============================
# LOAD MODEL (5090 OPTIMIZED)
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

    log("Loading model on RTX 5090...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # BF16 for 5090 (better than FP16)
        local_files_only=True,
        attn_implementation="flash_attention_2"  # If model supports it
    )

    model.eval()
    
    # Compile model for extra speed (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        log("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Warm up with realistic inference
    log("Warming up RTX 5090...")
    dummy_img = Image.new('RGB', (2048, 1440), color='white')
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Extract all text"}
        ]
    }]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[dummy_img], return_tensors="pt").to(DEVICE)
    
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = model.generate(**inputs, max_new_tokens=50)
    
    # Clear cache after warmup
    torch.cuda.empty_cache()
    
    # Print GPU stats
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}")
        log(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
        log(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    log("RTX 5090 ready - optimized for speed")


# ===============================
# OCR IMAGE (5090 OPTIMIZED)
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

    # Use automatic mixed precision for speed
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,  # Greedy for consistency
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
# HANDLER (5090 OPTIMIZED)
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

        # Optional parameters
        max_tokens = input_data.get("max_tokens", 1500)
        
        # Decode and process
        image = decode_image(input_data["image"])
        text = ocr_image(image, max_tokens=max_tokens)

        # Optional: return GPU stats for monitoring
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
log("="*50)
log("RTX 5090 OCR HANDLER")
log("="*50)
load_model()
log("Handler ready for lightning-fast inference!")

runpod.serverless.start({
    "handler": handler
})
