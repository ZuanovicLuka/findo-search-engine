from transformers import BartForConditionalGeneration, BartTokenizer


class Summarizer:
    def __init__(
        self,
        model_name="facebook/bart-large-cnn",
        max_input_length=1024,
        max_summary_length=400,
        min_summary_length=100,
        num_beams=4,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_summary_length = max_summary_length
        self.min_summary_length = min_summary_length
        self.num_beams = num_beams

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text: str):
        """Generate a summary for the given text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.max_input_length,
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.max_summary_length,
            min_length=self.min_summary_length,
            length_penalty=2.0,
            num_beams=self.num_beams,
            early_stopping=True,
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
