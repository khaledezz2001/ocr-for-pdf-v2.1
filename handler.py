import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_bytes

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
MAX_PAGES = 20

processor = None
model = None

# ===============================
# RTX 4090 OPTIMIZATIONS
# ===============================
if torch.cuda.is_available():
    # Enable TF32 for faster computation on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels
    
    # Set memory allocation strategy
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


# ===============================
# IMAGE DECODING (OPTIMIZED)
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    
    # Optimized resize for speed
    target_width = 1280  # Reduced from 1600 for faster processing
    scale = target_width / img.width
    new_height = int(img.height * scale)
    
    # LANCZOS is faster than BICUBIC with similar quality
    img = img.resize((target_width, new_height), Image.LANCZOS)
    return img


def decode_pdf(b64):
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes,
        dpi=120,  # Reduced from 150 for faster processing
        fmt="png",
        thread_count=6,  # Increased for better CPU utilization
        use_pdftocairo=True  # Faster rendering
    )
    return images[:MAX_PAGES]


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
        torch_dtype=torch.float16,  # Always use FP16 on GPU
        local_files_only=True,
        low_cpu_mem_usage=True  # Faster loading
    )

    model.eval()
    
    # Compile model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile') and DEVICE == "cuda":
        log("Compiling model with torch.compile...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log("Model compiled successfully")
        except Exception as e:
            log(f"Compilation failed: {e}, using eager mode")
    
    log("RolmOCR model loaded and ready")


# ===============================
# OCR ONE PAGE (OPTIMIZED)
# ===============================
@torch.inference_mode()  # Decorator for cleaner code
def ocr_page(image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Extract all text from this document including headers, "
                        "tables, footers, numbers, and special characters."
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
        return_tensors="pt",
        padding=True
    ).to(DEVICE, non_blocking=True)  # Non-blocking transfer

    # Optimized generation parameters
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,  # Reduced from 1200
        temperature=0.1,
        do_sample=False,  # Greedy decoding is faster
        num_beams=1,  # Disable beam search for speed
        use_cache=True,  # Enable KV cache
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    # Clean up the response
    if "assistant" in decoded:
        decoded = decoded.split("assistant", 1)[-1]

    decoded = decoded.strip()
    
    # Remove system/user prefixes
    for prefix in ["system\n", "user\n", "."]:
        if decoded.startswith(prefix):
            decoded = decoded.split("\n", 1)[-1].strip()

    return decoded


# ===============================
# BATCH PROCESSING (NEW)
# ===============================
@torch.inference_mode()
def ocr_batch(images: list[Image.Image]) -> list[str]:
    """Process multiple pages in a single batch for better GPU utilization"""
    if len(images) == 1:
        return [ocr_page(images[0])]
    
    # For small batches, process together
    batch_size = min(4, len(images))  # RTX 4090 can handle 4 pages
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        messages_list = []
        for _ in batch:
            messages_list.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "Extract all text from this document including headers, tables, footers, numbers, and special characters."
                        }
                    ]
                }
            ])
        
        prompts = [processor.apply_chat_template(m, add_generation_prompt=True) for m in messages_list]
        
        inputs = processor(
            text=prompts,
            images=batch,
            return_tensors="pt",
            padding=True
        ).to(DEVICE, non_blocking=True)
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        
        decoded_batch = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        for decoded in decoded_batch:
            if "assistant" in decoded:
                decoded = decoded.split("assistant", 1)[-1]
            decoded = decoded.strip()
            for prefix in ["system\n", "user\n", "."]:
                if decoded.startswith(prefix):
                    decoded = decoded.split("\n", 1)[-1].strip()
            results.append(decoded)
    
    return results


# ===============================
# HANDLER
# ===============================
def handler(event):
    load_model()

    PREFIX = (
        "Return the COMPLETE plain text of this document from top to bottom, "
        "including headers, tables, footers, bank details, signatures, stamps, "
        "emails, phone numbers, and all numbers exactly as written.\nassistant\n"
    )

    try:
        if "image" in event["input"]:
            pages = [decode_image(event["input"]["image"])]
        elif "file" in event["input"]:
            pages = decode_pdf(event["input"]["file"])
        else:
            return {
                "status": "error",
                "message": "Missing image or file"
            }

        # Use batch processing for better performance
        if len(pages) > 1:
            texts = ocr_batch(pages)
            extracted_pages = [
                {"page": i, "text": text.replace(PREFIX, "", 1)}
                for i, text in enumerate(texts, start=1)
            ]
        else:
            text = ocr_page(pages[0])
            extracted_pages = [{"page": 1, "text": text.replace(PREFIX, "", 1)}]

        # Clear cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "total_pages": len(extracted_pages),
            "pages": extracted_pages
        }
    
    except Exception as e:
        log(f"Error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


# ===============================
# PRELOAD & WARMUP
# ===============================
log("Preloading model...")
load_model()

# Warmup run for optimal performance
if torch.cuda.is_available():
    log("Running warmup...")
    dummy_image = Image.new('RGB', (1280, 1600), color='white')
    try:
        _ = ocr_page(dummy_image)
        torch.cuda.empty_cache()
        log("Warmup complete")
    except Exception as e:
        log(f"Warmup failed: {e}")

runpod.serverless.start({
    "handler": handler
})
