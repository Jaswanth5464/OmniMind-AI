#Supported file types
SUPPORTED_TEXT = {"pdf","txt","docx","pptx","csv","xls","xlsx","py","java"}
SUPPORTED_IMG  = {"png","jpg","jpeg"}
JUNK_EXTS      = {"exe","sys","dll","log","tmp","ini"}

#Models
EMBED_MODEL   = "BAAI/bge-base-en-v1.5"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
SUMMARIZER    = "facebook/bart-large-cnn"

#Text processing
TAG_MAX       = 5
CHUNK_WORDS   = 512
CHUNK_OVERLAP = 64
