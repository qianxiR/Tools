from transformers import pipeline

pipe = pipeline("zero-shot-image-classification", model="google/siglip2-base-patch16-224")
pipe(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png",
    candidate_labels=["animals", "humans", "landscape"],
)