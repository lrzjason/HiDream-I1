from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
    
)

hidream_dir = "F:/HiDream-I1/hidream_models/full"

tokenizer = CLIPTokenizer.from_pretrained(hidream_dir, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(hidream_dir, subfolder="tokenizer_2")
tokenizer_3 = T5Tokenizer.from_pretrained(hidream_dir, subfolder="tokenizer_3")

print("test")