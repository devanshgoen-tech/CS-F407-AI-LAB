"""HuggingFace Transformers demos.

Worksheet: AI_lab_transformers.ipynb.

Mirrors the three hands-on blocks:

  * Encoder-decoder (seq2seq) translation with T5
  * Decoder-only text generation with GPT-2
  * Encoder-only sentiment analysis with BERT (sentiment-analysis pipeline)

Weights are downloaded on first run and cached by HuggingFace.  The
script is safe to run offline afterwards.

    pip install "transformers[torch]"
    python3 hf_demos.py
"""

from __future__ import annotations


def demo_encoder_decoder_translation() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print("=== Encoder-decoder: T5 translation (English -> French) ===")
    tok = AutoTokenizer.from_pretrained("google-t5/t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
    prompt = "translate English to French: Let's go have some wine, cause that's how we shine"
    inputs = tok(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=30)
    print("  prompt:", prompt)
    print("  model :", tok.decode(output[0], skip_special_tokens=True))


def demo_decoder_only_generation() -> None:
    from transformers import pipeline

    print("\n=== Decoder-only: GPT-2 next-token generation ===")
    generator = pipeline("text-generation", model="gpt2")
    prompt = "The future of AI is"
    out = generator(prompt, max_length=30, num_return_sequences=1)
    print("  prompt:", prompt)
    print("  model :", out[0]["generated_text"])


def demo_encoder_only_sentiment() -> None:
    from transformers import pipeline

    print("\n=== Encoder-only: BERT sentiment classification ===")
    classifier = pipeline("sentiment-analysis")
    sentences = [
        "I love machine learning",
        "This homework is frustrating and endless",
        "The movie was okay I guess",
    ]
    for s in sentences:
        print(f"  '{s}' -> {classifier(s)}")


if __name__ == "__main__":
    demo_encoder_decoder_translation()
    demo_decoder_only_generation()
    demo_encoder_only_sentiment()
