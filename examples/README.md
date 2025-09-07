# NLP Examples - Hands-On Practice

These Jupyter notebooks provide practical implementations of concepts from our guides.

## Learning Order

### Beginners (Start Here)
1. **[preprocessing.ipynb](./preprocessing.ipynb)** - Clean and prepare text data
2. **[sentiment_analysis.ipynb](./sentiment_analysis.ipynb)** - Your first NLP model
3. **[text_classification.ipynb](./text_classification.ipynb)** - Categorize text automatically

### Intermediate
4. **[ner.ipynb](./ner.ipynb)** - Extract entities from text
5. **[word_embeddings.ipynb](./word_embeddings.ipynb)** - Understanding word vectors

## Concept Mapping

| Notebook | README.md Sections | Technical Guide Sections |
|----------|-------------------|-------------------------|
| preprocessing.ipynb | #1 (Tokenization), #2 (Stop Words), #3 (Stemming & Lemmatization) | Text Preprocessing |
| sentiment_analysis.ipynb | #6 (Sentiment Analysis) | Sentiment Analysis, Evaluation Metrics |
| text_classification.ipynb | #7 (Text Classification) | Text Classification, Machine Learning |
| ner.ipynb | #5 (Named Entity Recognition) | Named Entity Recognition |
| word_embeddings.ipynb | #15 (Word Embeddings) | Word Embeddings |

## Getting Started

### Prerequisites
- Basic Python knowledge
- Jupyter Notebook or JupyterLab installed

### Installation
1. Install required packages:
```bash
pip install numpy pandas scikit-learn nltk spacy textblob jupyter matplotlib seaborn
```

2. Download NLTK data (run in Python):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

3. For spaCy (used in NER):
```bash
python -m spacy download en_core_web_sm
```

4. Start Jupyter:
```bash
jupyter notebook
```

## 📖 How to Use These Notebooks

Each notebook follows this structure:
- **Introduction** - Links back to concepts in the main guides
- **Learning Objectives** - What you'll accomplish
- **Theory Recap** - Brief review of key concepts
- **Practical Examples** - Step-by-step implementations
- **Exercises** - Practice problems to test your understanding
- **Next Steps** - Where to go next in your learning journey

## Back to Main Guides

- **[Beginner Guide](../README.md)** - Concepts explained simply
- **[Technical Guide](../nlp-technical-guide.md)** - In-depth implementations and theory

## Tips for Success

1. **Start with preprocessing.ipynb** - It covers fundamental text preparation techniques
2. **Run code as you read** - Don't just read, experiment!
3. **Try the exercises** - They reinforce your learning
4. **Modify the examples** - Use your own text data
5. **Ask questions** - If something isn't clear, research it further

## Contributing

Found an error or want to add an example? We welcome contributions!
- Check for typos or code bugs
- Suggest additional examples
- Share interesting datasets
- Improve explanations

Happy learning!
