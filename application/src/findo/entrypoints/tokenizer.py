import re

from nltk.stem.snowball import SnowballStemmer


class Tokenizer:
    """Configurable text tokenizer for preprocessing and normalization"""

    def __init__(self, args):
        """Initialize tokenizer options (stopwords, stemming, regexes)"""
        self.args = args

        self.clean_words = re.compile(r"[^a-zA-Z0-9\s]+")

        if args.remove_urls:
            self.url_regex = re.compile(r"http[s]?://\S+|www\.\S+")
        if args.separate_alphanumeric:
            self.alpha_num_regex = re.compile(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])")

        if args.stopwords:
            with open(args.stopwords, encoding="utf-8") as f:
                self.stopwords = set(f.read().splitlines())
        else:
            self.stopwords = None

        self.stemmer = None
        if args.stemmer:
            self.stemmer = SnowballStemmer("portuguese")

    def tokenize(self, text):
        """Convert text into normalized tokens based on configuration"""
        if self.args.lowercase:
            text = text.lower()
        if self.args.remove_urls:
            text = self.url_regex.sub(" URL ", text)

        text = self.clean_words.sub("", text)

        if self.args.separate_alphanumeric:
            text = self.alpha_num_regex.sub(" ", text)

        tokens = text.split()
        result = []

        for token in tokens:
            if self.stopwords and token in self.stopwords:
                token = "<STOPWORD>"
            elif self.args.remove_numbers and token.isdigit():
                token = "<NUMBER>"
            if self.args.min_token_length > 0 and len(token) < self.args.min_token_length:
                token = "<SHORT-TOKEN>"

            if self.stemmer and token not in {
                "<STOPWORD>",
                "<NUMBER>",
                "<SHORT-TOKEN>",
                "<URL>",
                "URL",
                "url",
            }:
                token = self.stemmer.stem(token)

            result.append(token)

        return result
