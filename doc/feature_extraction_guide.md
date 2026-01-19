# Feature Extraction: Turning Raw Data into Numbers

## What is Feature Extraction?

**Feature extraction** is the process of converting raw data (text, images, audio) into numerical values that machine learning models can understand.

Think of it like translation:
- **Raw data:** "Call me back later" (human language)
- **Features:** `[0.0, 0.23, 0.0, 0.45, 0.12, ...]` (machine language)

## Why is Feature Extraction Necessary?

Machine learning models are just math equations. They can only work with numbers, not:
- Text strings
- Images
- Audio files
- Categories like "red", "blue", "green"

Feature extraction bridges this gap.

## Feature Extraction vs. Feature Engineering

| Term | What It Does | Example |
|------|--------------|---------|
| **Feature Extraction** | Converts non-numeric data to numbers | Text → TF-IDF vectors |
| **Feature Engineering** | Creates new features from existing ones | Message → Message Length |

Both are important, and they often happen together.

---

## Common Feature Extraction Techniques

### 1. For Text Data

#### Bag of Words (CountVectorizer)
Counts how many times each word appears.

**Example:**
```
Document 1: "I love cats"
Document 2: "I love dogs"
```

| Word | Doc 1 | Doc 2 |
|------|-------|-------|
| love | 1 | 1 |
| cats | 1 | 0 |
| dogs | 0 | 1 |

**Note:** With scikit-learn's default settings, `CountVectorizer` ignores 1-character tokens (like `"I"`), which is why it does not appear in the extracted feature list below.

**Code:**
```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love cats", "I love dogs"]
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
# ['cats', 'dogs', 'love']

print(features.toarray())
# [[1, 0, 1],
#  [0, 1, 1]]
```

**Pros:** Simple, easy to understand  
**Cons:** Common words like "the" dominate; ignores word importance

---

#### TF-IDF (Term Frequency - Inverse Document Frequency)
Weighs words by importance. Common words get lower scores; rare words get higher scores.

**The Idea:**
- **TF (Term Frequency):** How often a word appears in *this* document
- **IDF (Inverse Document Frequency):** How rare the word is across *all* documents

**Example:**
```
Document 1: "the cat sat on the mat"
Document 2: "the dog sat on the rug"
```

- "the" appears everywhere → Low weight
- "cat" only in Doc 1 → Higher weight
- "dog" only in Doc 2 → Higher weight

**Code:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["the cat sat on the mat", "the dog sat on the rug"]
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(texts)

print(features.toarray())
```

**Pros:** Better than raw counts; reduces impact of common words  
**Cons:** Still ignores word order and meaning

---

#### Word Embeddings (Word2Vec, GloVe)
Represents words as dense vectors that capture meaning. Similar words have similar vectors.

**Example:**
```
"king"  → [0.2, 0.5, 0.1, ...]
"queen" → [0.21, 0.48, 0.09, ...]  (similar!)
"apple" → [0.8, 0.1, 0.7, ...]     (different)
```

**Often-cited example (not guaranteed):**
```
king - man + woman ≈ queen
```

This kind of relationship can show up in some embedding models, but it depends on the data, training method, and the specific embedding.

**Pros:** Captures semantic meaning  
**Cons:** More complex; requires pre-trained models or large datasets

---

### 2. For Categorical Data

#### One-Hot Encoding
Converts categories into binary columns (0 or 1).

**Example:**
```
Color: ["red", "blue", "green", "red"]
```

| red | blue | green |
|-----|------|-------|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |
| 1 | 0 | 0 |

**Code:**
```python
import pandas as pd

df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
one_hot = pd.get_dummies(df["color"])
print(one_hot)
```

**Pros:** Simple, works well for most cases  
**Cons:** Creates many columns if there are many categories

---

#### Label Encoding
Assigns a number to each category.

**Example:**
```
["small", "medium", "large"] → [0, 1, 2]
```

**Code:**
```python
from sklearn.preprocessing import LabelEncoder

sizes = ["small", "medium", "large", "small"]
encoder = LabelEncoder()
encoded = encoder.fit_transform(sizes)
print(encoded)  # [2, 1, 0, 2]
```

**Important note:** For scikit-learn, `LabelEncoder` is mainly intended for encoding the **target label** (`y`).
For **input features** (`X`), prefer:
- `OneHotEncoder` for categories with *no* natural order (e.g., colors)
- `OrdinalEncoder` for categories with a natural order that you define (e.g., small < medium < large)

**Pros:** Compact; can be appropriate for ordinal features  
**Cons:** Can mislead models if the category order is not truly meaningful (the model may assume 2 > 1 > 0 has numeric meaning)

---

### 3. For Images

#### Pixel Values
Flatten the image into a long list of pixel intensities.

**Example:** A 28×28 grayscale image becomes 784 numbers (0-255).

```python
# Pseudo-code
image = load_image("digit.png")  # Shape: (28, 28)
features = image.flatten()        # Shape: (784,)
```

**Pros:** Simple  
**Cons:** Loses spatial information; doesn't scale well

---

#### Convolutional Neural Networks (CNNs)
Deep learning models that automatically extract features from images (edges, shapes, objects).

**Pros:** State-of-the-art for image tasks  
**Cons:** Requires lots of data and compute

---

### 4. For Numerical Data

#### Normalization (Min-Max Scaling)
Scales values to a 0-1 range.

**Formula:** `(x - min) / (max - min)`

**Example:**
```
Ages: [20, 40, 60] → [0.0, 0.5, 1.0]
```

**Code:**
```python
from sklearn.preprocessing import MinMaxScaler

ages = [[20], [40], [60]]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(ages)
print(scaled)  # [[0.0], [0.5], [1.0]]
```

---

#### Standardization (Z-Score)
Centers data around 0 with standard deviation of 1.

**Formula:** `(x - mean) / std`

**Code:**
```python
from sklearn.preprocessing import StandardScaler

values = [[10], [20], [30]]
scaler = StandardScaler()
scaled = scaler.fit_transform(values)
print(scaled)  # [[-1.22], [0.0], [1.22]]
```

---

## Quick Reference: Which Technique to Use?

| Data Type | Recommended Technique |
|-----------|----------------------|
| Text (simple) | TF-IDF |
| Text (advanced) | Word Embeddings |
| Categories (no order) | One-Hot Encoding |
| Categories (ordered) | Ordinal Encoding (e.g., `OrdinalEncoder`) |
| Numbers (different scales) | Standardization |
| Numbers (need 0-1 range) | Normalization |
| Images (simple) | Pixel Flattening |
| Images (advanced) | CNN Features |

---

## Complete Example: Text Classification Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. Load data
df = pd.read_csv("spam.csv")
X = df["message"]
y = df["label"]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_features = vectorizer.fit_transform(X_train)  # Learn + transform
X_test_features = vectorizer.transform(X_test)        # Transform only

# 4. Train model
model = LinearSVC()
model.fit(X_train_features, y_train)

# 5. Evaluate
predictions = model.predict(X_test_features)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
```

---

## Key Takeaways

1. **Feature extraction converts raw data into numbers** that models can process.
2. **Different data types need different techniques** (text, categories, images, numbers).
3. **TF-IDF is the go-to method for text classification** (simple and effective).
4. **Always fit on training data only**, then transform both train and test sets.
5. **Feature extraction quality directly impacts model performance**—garbage in, garbage out.

---

## Summary Table

| Technique | Input | Output | Use Case |
|-----------|-------|--------|----------|
| CountVectorizer | Text | Sparse matrix | Simple text tasks |
| TfidfVectorizer | Text | Sparse matrix | Text classification |
| Word2Vec/GloVe | Text | Dense vectors | Semantic similarity |
| One-Hot Encoding | Categories | Binary columns | Nominal categories |
| OrdinalEncoder (features) / LabelEncoder (target) | Categories | Integers | Ordinal features / target labels |
| StandardScaler | Numbers | Centered numbers | Most ML models |
| MinMaxScaler | Numbers | 0-1 range | Neural networks |
