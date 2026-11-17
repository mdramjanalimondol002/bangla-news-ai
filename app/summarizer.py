import os
import re
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


class BanglaNewsSummarizer:
    """Abstractive Bengali news summarizer powered by Google Gemini with extractive fallback."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

    def summarize(self, article_text: str, language: str = "Bengali") -> dict:
        """Generate a concise summary and bullet-point key takeaways."""
        text = article_text.strip()
        if not text:
            return {"summary": "অনুগ্রহ করে একটি সংবাদ অনুচ্ছেদ প্রদান করুন।", "mode": "empty"}

        if self.api_key and self.api_key != "YOUR_COPIED_KEY_HERE":
            try:
                client = genai.Client(api_key=self.api_key)
                prompt = (
                    f"You are an expert Bengali news editor. Read the following Bengali news article "
                    f"and provide:\n"
                    f"1. A crisp 2-sentence executive summary in {language}.\n"
                    f"2. Three concise bullet-point key takeaways (who, what, where/impact) in {language}.\n\n"
                    f"Article:\n{text}"
                )
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
                return {"summary": response.text.strip(), "mode": f"Gemini ({GEMINI_MODEL})"}
            except Exception:
                pass

        # Extractive fallback when running offline or without an API key
        sentences = [s.strip() for s in re.split(r"[।.!?]+", text) if s.strip()]
        lead_sentences = "। ".join(sentences[:2]) + ("।" if sentences else "")
        bullets = "\n".join([f"• {s}।" for s in sentences[:3]])
        fallback_output = (
            f"**সংক্ষিপ্ত সারসংক্ষেপ (Extractive Summary):**\n{lead_sentences}\n\n"
            f"**মূল তথ্যসমূহ (Key Highlights):**\n{bullets}"
        )
        return {"summary": fallback_output, "mode": "Offline Extractive Fallback"}