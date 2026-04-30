import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Step 1: Knowledge base ---
docs = [
    "The Sun is the center of the solar system.",
    "The Earth revolves around the Sun once every 365 days.",
    "Mars is known as the red planet.",
    "Jupiter is the largest planet in the solar system."
]

# --- Step 2: Indexing (TF-IDF embeddings) ---
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(docs)

# --- Step 3: Retrieval ---
def retrieve(query, top_k=2):
    query_vec = vectorizer.transform([query])
    scores = (doc_vectors @ query_vec.T).toarray().ravel()
    top_ids = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], scores[i]) for i in top_ids]

# --- Step 4: "Generation" (simple template) ---
def rag_answer(query):
    retrieved = retrieve(query)
    context = " ".join([doc for doc, _ in retrieved])
    return f"Q: {query}\nA: Based on retrieved info: {context}"

# --- Example ---
print(rag_answer("Which planet is largest?"))
