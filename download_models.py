import os
import ssl
import sys

# 🛠️ SSL FIX: Only used if we actually need to download
def enable_ssl_fix():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
    except AttributeError:
        pass

print("🚀 Checking AI Models...")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
    from langchain_huggingface import HuggingFaceEmbeddings
    
    def smart_download(name, loader_func, *args, **kwargs):
        print(f"\n--- Checking {name} ---")
        try:
            # First, try to load strictly from local files
            loader_func(*args, **kwargs, local_files_only=True)
            print(f"✅ {name} already exists on this PC. Skipping download.")
        except Exception:
            # If local fails, enable the SSL fix and download
            print(f"🔍 {name} not found. Downloading now (requires internet)...")
            enable_ssl_fix()
            loader_func(*args, **kwargs, local_files_only=False)
            print(f"✅ {name} downloaded successfully.")

    # 1. BLIP
    smart_download("Image Model (BLIP Processor)", BlipProcessor.from_pretrained, "Salesforce/blip-image-captioning-base")
    smart_download("Image Model (BLIP Model)", BlipForConditionalGeneration.from_pretrained, "Salesforce/blip-image-captioning-base")

    # 2. Summarizer
    smart_download("Summarizer (Tokenizer)", AutoTokenizer.from_pretrained, "facebook/bart-large-cnn")
    smart_download("Summarizer (Model)", AutoModelForSeq2SeqLM.from_pretrained, "facebook/bart-large-cnn")

    # 3. Embeddings
    print("\n--- Checking Search Model (BGE) ---")
    try:
        HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"local_files_only": True})
        print("✅ Search Model already exists. Skipping.")
    except Exception:
        print("🔍 Search Model not found. Downloading now...")
        enable_ssl_fix()
        HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"local_files_only": False})
        print("✅ Search Model downloaded.")

    print("\n" + "="*50)
    print("✨ ALL MODELS ARE READY!")
    print("🔒 You can now go 100% OFFLINE and run 'python main.py'.")
    print("="*50)

except Exception as e:
    print(f"\n❌ AN ERROR OCCURRED: {e}")
