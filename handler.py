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
# HALLUCINATION DETECTION
# ===============================
def is_hallucinated_output(text: str) -> bool:
    """Detect if the OCR output is hallucinated/garbage"""
    if not text or len(text.strip()) < 10:
        return True
    
    # Check for repetitive table patterns (like page 13)
    lines = text.strip().split('\n')
    if len(lines) > 20:
        # Check if most lines are identical or very similar
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(unique_lines) < 3:  # Too repetitive
            return True
    
    # Check for excessive markdown tables with no content
    table_markers = text.count('|')
    if table_markers > 50 and len(text.replace('|', '').replace('\n', '').strip()) < 50:
        return True
    
    # Check for only special characters
    alphanumeric_chars = sum(c.isalnum() for c in text)
    if alphanumeric_chars < 10:
        return True
    
    return False


# ===============================
# IMAGE DECODING (ACCURACY OPTIMIZED)
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    
    # Higher resolution for better accuracy
    target_width = 1920  # Increased for better text recognition
    scale = target_width / img.width
    new_height = int(img.height * scale)
    
    # BICUBIC for better quality
    img = img.resize((target_width, new_height), Image.BICUBIC)
    return img


def decode_pdf(b64):
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes,
        dpi=200,  # Increased for better quality
        fmt="png",
        thread_count=4,
        use_pdftocairo=True
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
# OCR ONE PAGE (ACCURACY OPTIMIZED)
# ===============================
@torch.inference_mode()
def ocr_page(image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "You are a professional OCR system. Extract ALL text from this document "
                        "EXACTLY as written. Include:\n"
                        "- All headers, titles, and sections\n"
                        "- All body text and paragraphs\n"
                        "- All tables with correct alignment\n"
                        "- All numbers, dates, and codes EXACTLY as shown\n"
                        "- All names, addresses, and contact information\n"
                        "- All signatures, stamps, and annotations\n"
                        "- Preserve original spelling and formatting\n"
                        "- Do NOT correct typos or translate anything\n"
                        "- Do NOT add interpretations or summaries\n"
                        "Return ONLY the extracted text, nothing else."
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
    ).to(DEVICE, non_blocking=True)

    # More conservative generation for accuracy
    output_ids = model.generate(
        **inputs,
        max_new_tokens=2048,  # Increased for long documents
        min_new_tokens=10,    # Ensure some output
        temperature=0.0,      # Deterministic for consistency
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.1,  # Reduce repetition
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False  # Keep original spacing
    )[0]

    # Clean up the response more carefully
    if "assistant" in decoded.lower():
        parts = decoded.lower().split("assistant")
        if len(parts) > 1:
            decoded = decoded[decoded.lower().index("assistant") + len("assistant"):]

    decoded = decoded.strip()
    
    # Remove ALL system/user/assistant prefixes more aggressively
    prefixes_to_remove = [
        "system\n", "user\n", "assistant\n",
        "System\n", "User\n", "Assistant\n",
        "SYSTEM\n", "USER\n", "ASSISTANT\n"
    ]
    
    for prefix in prefixes_to_remove:
        if decoded.lower().startswith(prefix.lower()):
            decoded = decoded[len(prefix):].strip()
    
    # Remove the extraction instruction if it appears
    instruction = "extract all text from this document including headers, tables, footers, numbers, and special characters."
    if decoded.lower().startswith(instruction):
        decoded = decoded[len(instruction):].strip()
    
    # Remove leading dots/colons only if they're artifacts
    if decoded.startswith((".\n", ":\n", ". ", ": ")):
        decoded = decoded.lstrip(".:").strip()

    return decoded


# ===============================
# BATCH PROCESSING (ACCURACY OPTIMIZED)
# ===============================
@torch.inference_mode()
def ocr_batch(images: list[Image.Image]) -> list[str]:
    """Process multiple pages - with accuracy priority"""
    if len(images) == 1:
        return [ocr_page(images[0])]
    
    # Smaller batches for better accuracy
    batch_size = min(2, len(images))  # Reduced to 2 for better quality
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
                            "text": (
                                "You are a professional OCR system. Extract ALL text from this document "
                                "EXACTLY as written. Include:\n"
                                "- All headers, titles, and sections\n"
                                "- All body text and paragraphs\n"
                                "- All tables with correct alignment\n"
                                "- All numbers, dates, and codes EXACTLY as shown\n"
                                "- All names, addresses, and contact information\n"
                                "- All signatures, stamps, and annotations\n"
                                "- Preserve original spelling and formatting\n"
                                "- Do NOT correct typos or translate anything\n"
                                "- Do NOT add interpretations or summaries\n"
                                "Return ONLY the extracted text, nothing else."
                            )
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
            max_new_tokens=2048,
            min_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        
        decoded_batch = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        for decoded in decoded_batch:
            if "assistant" in decoded.lower():
                parts = decoded.lower().split("assistant")
                if len(parts) > 1:
                    decoded = decoded[decoded.lower().index("assistant") + len("assistant"):]
            
            decoded = decoded.strip()
            
            # Remove ALL system/user/assistant prefixes more aggressively
            prefixes_to_remove = [
                "system\n", "user\n", "assistant\n",
                "System\n", "User\n", "Assistant\n",
                "SYSTEM\n", "USER\n", "ASSISTANT\n"
            ]
            
            for prefix in prefixes_to_remove:
                if decoded.lower().startswith(prefix.lower()):
                    decoded = decoded[len(prefix):].strip()
            
            # Remove the extraction instruction if it appears
            instruction = "extract all text from this document including headers, tables, footers, numbers, and special characters."
            if decoded.lower().startswith(instruction):
                decoded = decoded[len(instruction):].strip()
            
            # Remove leading dots/colons only if they're artifacts
            if decoded.startswith((".\n", ":\n", ". ", ": ")):
                decoded = decoded.lstrip(".:").strip()
            
            results.append(decoded)
    
    return results


# ===============================
# HANDLER
# ===============================
def handler(event):
    load_model()

    # Prefixes that might appear in output and should be removed
    PREFIXES_TO_REMOVE = [
        "user\nExtract all text from this document including headers, tables, footers, numbers, and special characters.\nassistant\n",
        "user\n",
        "assistant\n",
        "system\n",
        "User\n",
        "Assistant\n",
        "System\n",
        "Extract all text from this document including headers, tables, footers, numbers, and special characters.\n",
        "Return the COMPLETE plain text of this document from top to bottom, including headers, tables, footers, bank details, signatures, stamps, emails, phone numbers, and all numbers exactly as written.\nassistant\n"
    ]

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
            extracted_pages = []
            for i, text in enumerate(texts, start=1):
                # Remove all known prefixes
                for prefix in PREFIXES_TO_REMOVE:
                    if text.startswith(prefix):
                        text = text[len(prefix):]
                        break  # Only remove first match
                
                text = text.strip()
                
                # Skip hallucinated/garbage pages
                if is_hallucinated_output(text):
                    log(f"Warning: Page {i} appears to be hallucinated, marking as empty")
                    text = "[Empty or unreadable page]"
                
                extracted_pages.append({"page": i, "text": text})
        else:
            text = ocr_page(pages[0])
            
            # Remove all known prefixes
            for prefix in PREFIXES_TO_REMOVE:
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            
            text = text.strip()
            
            # Check for hallucination
            if is_hallucinated_output(text):
                log("Warning: Page appears to be hallucinated, marking as empty")
                text = "[Empty or unreadable page]"
            
            extracted_pages = [{"page": 1, "text": text}]

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
    dummy_image = Image.new('RGB', (1920, 1600), color='white')  # Match resolution
    try:
        _ = ocr_page(dummy_image)
        torch.cuda.empty_cache()
        log("Warmup complete")
    except Exception as e:
        log(f"Warmup failed: {e}")

runpod.serverless.start({
    "handler": handler
})
