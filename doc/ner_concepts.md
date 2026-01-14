# Named Entity Recognition (NER) Concepts

This document explains the core concepts used in Named Entity Recognition (NER), specifically within the context of the spaCy NLP library.

## 1. Named Entity Recognition (NER)
**NER** is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories. Common categories include person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

**Goal:** To turn unstructured text into structured data by identifying "who," "what," "where," and "when."

---

## 2. Core Components

### Token
A **Token** is the fundamental unit of processing in NLP. It usually represents a single word, punctuation mark, or symbol.

*   **Analogy:** The "bricks" that make up the sentence.
*   **Example:** In the sentence `"Tesla created it."`, the tokens are `["Tesla", "created", "it", "."]`.

### Span
A **Span** is a continuous sequence of one or more tokens (a slice of a document).

*   **Analogy:** A "selection" or "highlighted text" within the document.
*   **Properties:** A span has a `start` token index and an `end` token index.
*   **Example:** In `"San Francisco is foggy"`, `doc[0:2]` corresponds to the span `"San Francisco"`.

### Entity (`ent`)
An **Entity** is a specific type of **Span** that has been identified as having real-world significance (a "Named Entity"). Even if an entity consists of multiple tokens (e.g., "New York City"), it is treated as a single object in the list of entities.

*   **Analogy:** A meaningful "structure" built from token bricks.
*   **Rule:** In standard spaCy, a token can belong to at most **one** entity. Overlapping entities are generally not allowed in the main `doc.ents` list.
*   **Example:** In `"Tim Cook works at Apple"`, there are two entities:
    1.  `"Tim Cook"` (comprising 2 tokens)
    2.  `"Apple"` (comprising 1 token)

### Label
A **Label** is the "tag" or category assigned to an Entity. It describes *what kind* of object the entity is.

*   **Analogy:** The "sticker" or "category tag" attached to the entity structure.
*   **Common Labels:**
    *   `PERSON`: People, including fictional.
    *   `ORG`: Companies, agencies, institutions.
    *   `GPE`: Geopolitical entities (countries, cities, states).
    *   `DATE`: Absolute or relative dates or periods.

---

## 3. Relationships Summary

| Concept | Description | Multiplicity |
| :--- | :--- | :--- |
| **Token** | The atomic unit (word). | 1 Token |
| **Span** | A slice of the doc. | 1 or more Tokens |
| **Entity** | A meaningful Span.| 1 Span (1+ Tokens) |
| **Label** | The type of the Entity. | 1 Label per Entity |

### Key Takeaway
*   **Token vs. Entity:** An **Entity** "wraps" one or more **Tokens**.
*   **Entity vs. Label:** The **Entity** is the object found in the text (e.g., "Tesla"); the **Label** is the category assigned to it (e.g., "ORG").
