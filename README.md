# "Findo" - Search Engine Project

This project implements a search engine called "Findo" with both a REST API for searching and a CLI interface for indexing. It's built with strict typing, linting, and memory constraints to ensure high-quality, efficient code.

## Dataset

This project requires `ptwiki-articles-with-redirects.arrow` (≈ 1.7GB).  
Due to GitHub file size limits, the dataset is not included in the repository.

Download it from Dropbox and place it in the `application/data/` folder:

- Download: https://www.dropbox.com/scl/fi/9c4rff1dmym5ud8aahn8r/ptwiki-articles-with-redirects.arrow?rlkey=cviaz6lfk3xrji9t4um4g87d3&e=1&st=5impxuks&dl=1
- Path: `application/data/ptwiki-articles-with-redirects.arrow`

## Prerequisites

- uv (install from https://docs.astral.sh/uv/getting-started/installation/)

## Setup

For this project's code to work, first you need to run the whole process of creating an index, this is a step by step what you have to do:

1. Install dependencies:

```bash
uv sync
```

2. The CLI indexer runs with memory monitoring enabled to enforce the 2GB memory limit::

```bash
uv run cli [arguments]
```

OR

```bash
uv run src/findo/entrypoints/cli [arguments]
```

FOR EXAMPLE

```bash
uv run cli data/ptwiki-articles-with-redirects.arrow --stopwords data/pt-stopwords.txt --no-stemmer
```

3. Creating a database:
   We converted the arrow file into the sqlite database (database.db) because of the faster fetch when trying to get the title and the content of a searched document. This is not a part of indexing process, so we placed the code for that convertion inside the database_creation.py which has to be run separately after the indexing is finished. The command for that is this:

```bash
uv run src/findo/entrypoints/database_creation.py
```

4. Run the Search API:

```bash
uv run uvicorn findo.entrypoints.asgi:app --reload
```

The whole task was devided in the following subtasks, which are explained in the next part:

## 1. PREPROCESSING

Our starting dataset was a ptwiki-articles-with-redirects.arrow file with a corpus of wikipedia documents. It contains aprox. 2.7M documents, each one containing these parts:

- Title: The title of the page
- Text: corpus of the page
- Out links: A list of links to other pages that this page contains
- Redirect: Indicates whether the page is a redirect to another page

But 57.46% of the documents are the ones that have empty text field. We cleared the dataset from these and converted the arrow file into the sqlite database (database.db) because of the faster fetch when trying to get the title and the content of a searched document, so in the end we are using 1.154.228 documents.

## 2. TOKENIZER

In order to transform the text into tokens, each text had to be split into the one word tokens, following some of the rules that are given by running the cli.py file with special arguments.
Some of the possibilites are for text to be cleared from the stopwords (they are inside of the data/pt-stopwords.txt), the numbers can be removed or not, if token is shorter than the defined lenght it is removed and URLs can also be removed or not.
One more possibility is to turn on the Portugese stemmer (We used nltk.stem.snowball).

## 3. INDEXER

After each text is tokenized, the list of tokens is sent to the indexer. The indexing process was performed using the Single-Pass In-Memory Indexing (SPIMI) algorithm, implemented in the SPIMIIndexer class.

### In-Memory index construction

For each document, tokens were counted to generate term frequencies. The document’s length and ID were stored for later retrieval, inside demo-index/doc_stats.jsonl. The inverted index was maintained as a dictionary mapping terms to lists of (doc_id, frequency) pairs. When the token count exceeded a defined threshold, the indexer flushed the data to disk as a sorted block (.jsonl format). This ensured that memory consumption stayed within the specified limit of 2000MB.

### Finalization and metadata

After all documents were processed, any remaining terms in memory were written to disk as the last block and global metadata (average document length, total tokens, total documents) were saved to the demo-index/metadata.json.

### Block Merging

All intermediate blocks were then merged into a single final inverted index (final_index.jsonl)
This merge step used a priority queue (heap) to efficiently merge terms from multiple sorted blocks, combined duplicate postings and applied a minimum term frequency filter and produced a compact and fully merged inverted index suitable for fast lookup.

### Term Offsets

A separate pass recorded byte offsets for each term in the final index (term_offsets.json), enabling direct random access during search, which made searching much faster.

## 4. SEARCHER

The document retrieval process is handled by the Searcher component, which utilizes the previously built final inverted index (final_index.jsonl) to efficiently locate and rank relevant documents.

### Initialization

At startup, the Searcher loads several essential resources:

- Inverted index (final_index.jsonl) for term-to-document mappings

- Term offsets (term_offsets.json) to allow random file access to specific terms

- Document statistics (doc_stats.jsonl) containing document lengths

- Metadata (metadata.json) with global corpus information (total docs, average length)

- Configuration (config_file.json) for consistent tokenization parameters

- SQLite database for retrieving document titles and full text content

All configuration and metadata files are parsed into structured namespaces for easy access.

### Query processing

When a query is received, it is first tokenized using the same rules as during indexing (lowercasing, stemming, filtering stopwords, numbers, and URLs) from config_file.json for consistent tokenization parameters.

### Postings retrieval

For each token in the query, its byte offset is retrieved from term_offsets.json. The Searcher seeks directly to that position in final_index.jsonl, loads the line, and extracts the postings list (document IDs with term frequencies). Results are cached in memory to speed up subsequent lookups.

### Ranking with BM25 algorithm

Documents are scored using the BM25 ranking algorithm, which balances term frequency, document length, and term rarity. For each document in the postings, the searcher computes a BM25 score using parameters k = 1.2 and b = 0.75 as said in the assigment instructions, but the parameters can also be changed inside the Searcher class. Document lengths are retrieved from doc_stats.jsonl. The final score is the sum of individual term contributions and the top results are then sorted by descending score. The number of results shown is the one that user entered in the web interface while searching.

### Result retrieval and output

The Searcher retrieves the title and content of the top-ranked documents from the SQLite database and displays them as search results. Performance statistics, such as search duration, number of results, and top document scores are printed for monitoring.

### Similar search

Another posibility is to search for documents similar to any of the documents shown in the result of the initial search. When user hovers over the specific document, button "Search Similar" appears. After clicking the button, searcher performs a similar document search. First it takes the content of the document from which the button was clicked, tokenizes the text and then it applies term distillation, keeping only the most informative tokens based on their tf–idf weight. Also, extremely frequent terms (appearing in >20% of documents) are ignored to reduce noise.

### Performance

The normal (query) search is very fast, taking always less than 1s, but the similar search is just a bit slower around 3s, because usually the documents are very large (with a lot of words) so the tokenization + search process for each token takes some time. That's why we implemented the term distillation, so if the search has to be faster you can just adjust the number of most informative tokens (by default 32 tokens are used).

## Gemini API Key

Some features in this project use the **Google Gemini API**. The API key is always loaded from an environment variable stored in a `.env` file located in the root of the **application** folder.

- Environment variable name: `GEMINI_API_KEY`
- Only **Gemini free-tier models** are used

To enable these features, create a `.env` file in the project root and add:

```env
GEMINI_API_KEY=your_api_key_here
```

## 5. NEURAL RERANKER

For this part of the assigment we implemented neural reranker with the goal of achieving better ranking of the documents on top of the list. We used our old BM25 algorithm from assigment 1 to retrive top K documents (generally we retrieved top 50 documents). After the documents are retrieved, we pass them to the neural relevance model with the query and calculate similarity score for each document. Documents are then sorted by the similarity/neural score and final ordering is achieved.

For the neural reranker we used **CrossEncoders** from **sentence_transformers** that are available on Hugging Face.CrossEncoders are models that take a query and a document as a single input pair and produce a relevance score based on their joint representation. The whole idea is to capture precise interactions between the query and document, making them particularly effective for reranking sets of candidate documents.

In order to see which model should we use for this task, detailed comparisons between speed and accuracy of the 7 different models have been made. We tested the speed by averaging execution time on 4 queries and accuracy by executing ranking on the TREC DL 2019 dataset (the dataset is too big to be put on the github, so if you want to run it contact me on this email and I will explain what you have to download: luka.zuanovic@ua.pt). All of the times and accuracy scores can be found in the **evaluation** folder. From the results, we have decided to go with the "unicamp-dl/mMiniLM-L6-v2-pt-v2" model, because it's fast enough while still being very accurate (also because it's a portugese trained model that suits the portugese wikipedia corpus).

## 6. ANSWER GENERATION

This component of the system adds a natural-language answer generation layer on top of the traditional search engine. While the searcher returns documents ranked by BM25 and reranked by a neural reranker, the user often prefers a direct answer to a question. To support this, we implemented a dedicated Answer Generator, powered by the Gemini API, responsible for:

- Detecting whether the user query is a question
- Producing a final natural-language answer using the top retrieved document or if the document does not contain the required information, generating an AI-produced answer explicitly identified as such

All answer-generation logic is implemented in **answer_generation.py**

### 6.1. System Overview

When the user submits a query, the pipeline is extended as follows:

1. Query correction suggestion (explained in part **3. FURTHER AI ENHANCEMENTS**)

2. Question detection via LLM

3. Document retrieval using the BM25 + neural reranker pipeline

4. Answer generation using the content of the highest-ranked document or just by the AI

### 6.2. Question Detection

Before attempting to generate an answer, the system must determine whether the query is a question in Portuguese.
So the query is given to the **gemini-2.5-flash** model and a prompt looks like this:

"Respond only with 'YES' or 'NO'. Is the following sentence a question in Portuguese?"

This logic is activated through the **/is_question endpoint inside search.py**
If the output is NO, the system bypasses answer generation entirely and simply displays the standard ranked results.

### 6.3. Answer Generation

If the query is classified as a question, the front-end automatically calls the **/generate_answer** endpoint.
The Answer Generator then constructs a RAG-style prompt inside the **generate_answer()** method in answer_generation.py containing:

1. The user’s question (query), so the model understands what must be answered

2. The full content of the top retrieved document

3. Strict behavioral instructions. The prompt enforces the following rules:

If the answer is present in the document, the model must respond using only document-based information. If the document does not contain the required information, the model should answer using its own knowledge but must explicitly append:
“(Resposta gerada por IA, informação não encontrada no documento)”

In the end the answer is displayed above the result list in the dedicated “Answer” panel of the UI.

## 7. FURTHER AI ENHANCEMENTS

### 7.1. Query corrector

One of the enhancements that could improve user experience is Query Corrector, responsible for automatically detecting and fixing spelling or minor grammatical errors in Portuguese search queries. Misspelled queries are a common issue in real-world search behavior and can significantly reduce retrieval quality, because BM25 and neural rerankers depend on correctly written terms to match documents effectively. The Query Corrector therefore acts as the first layer in the search pipeline, ensuring that subsequent retrieval stages operate on clean and meaningful input.

The Query Corrector relies on the **Gemini 2.5 Flash Lite** model. This model is lightweight, fast, and inexpensive to call, making it suitable for real-time correction of short queries. When the user submits a query, the system sends a specialized prompt to Gemini instructing it to correct only spelling and light grammatical mistakes, keep the meaning unchanged and return only the corrected query, without explanations. The prompt is intentionally restrictive to avoid mistakes and to work like regular autocorrect system.

Every submitted query triggers a call to the **/correct_query** endpoint. If the corrected query differs from the original, the UI displays a “Did you mean…?” suggestion. Even if LLM detected that the query contains mistakes, search is still executed with a mistaken query. The user can click the suggested query, and the system automatically re-executes the search with the corrected version.

This integration ensures fewer potential “no results” scenarios caused by typos, better results in BM25 retrieval and improved neural reranker performance.

We tested some of the autocorrect models but it was hard to find some that are trained on portuguese so we decided to go with the Gemini like stated before.

### 7.2. Semantic snippet extraction

Another important enhancement is Semantic snippet extraction, whose goal is to present the user with the most relevant parts of each retrieved document instead of showing a naive fixed-length text fragment. Rather than extracting the first N characters of a document, our system identifies the sentences that are semantically closest to the user query and builds a compact, informative snippet from them.

This component is implemented directly inside the search pipeline and reuses the same neural reranking model used for document reranking. After a document is selected as part of the final ranked results, its full content is first split into sentences using Portuguese-specific punctuation heuristics. To keep the process efficient, only the first 50 sentences are considered. Each sentence is then paired with the original query and scored using the neural reranker, which computes a relevance score capturing deep semantic interactions between the query and the sentence. The top-k highest scoring sentences (by default with k = 3) are selected, reordered according to their original position in the text to preserve readability, and concatenated into a single snippet. Finally, the snippet length is capped at 400 characters to ensure concise display in the UI.

By leveraging the same cross-encoder model used for reranking documents, semantic snippet extraction ensures strong alignment between document ranking and snippet quality. The generated snippet is shown by default for each result, giving the user an immediate overview of the most relevant content. A "Show more" button allows the user to expand the result and view the complete document text, and once expanded, a "Show less" option enables switching back to the semantic snippet. This interaction improves usability while keeping the result list clean and easy to scan.

### 7.3. Summarize

Since some of the documents are pretty large, we thought it would be useful feature to be able to summarize the text. We tried to use some models that are specialized for summarization and found that for our documents "facebook/bart-large-cnn" model works the best.

We chose this model because BART (Bidirectional and Auto-Regressive Transformers) is specifically designed for sequence-to-sequence tasks such as text summarization. The bart-large-cnn variant has been fine-tuned on the CNN/DailyMail summarization dataset, making it highly effective at producing concise, coherent, and high-quality summaries. Even though this model was fine-tuned on english based summaries, it produced very good results on portuguese documents. Compared to other models we tested, BART provided summaries that were very informative, while also handling longer input texts, making it suitable for our summarization pipeline.

We tried using Gemini for the sumarization task and it showed pretty good results as well. However, BART made it easier to control the length of the summaries and since we have limited api calls to the Gemini we decided to go for this solution.

Also to make the user experience as smooth as possible, each computed summary is saved so that the users can toggle between real text and summaries immediately without waiting again for computing it by the model.
