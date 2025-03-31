
# Netflix Movie Recommendation System using LLaMA and FAISS

This project builds a content-based recommendation system for Netflix movies using LLaMA embeddings and FAISS vector similarity search. It creates vector representations of movie metadata and finds the most similar content based on user queries.

---

## Requirements

Make sure you have Python 3.10 or above installed.

### Create a virtual environment
```bash
python3 -m venv ollamavenv
source ollamavenv/bin/activate
```

### Install required packages
```bash
pip install numpy pandas faiss-cpu requests
```

---

## Step 1: Download and Run LLaMA Locally

You need to install [Ollama](https://ollama.com/) to run LLaMA models locally.

### Install Ollama

Download and install it from: https://ollama.com/download

### Pull the LLaMA3 model

```bash
ollama pull llama3
```

### Run the Ollama server

Ensure your local Ollama server is running (usually starts on http://localhost:11434)

---

## Step 2: Prepare Your Dataset

Download or use `netflix_titles.csv` with the following columns:

- `title`, `type`, `cast`, `director`, `release_year`, `listed_in`, `description`

Make sure your CSV file is in the same folder as your code or update the path accordingly.

---

## Step 3: Full Code to Run

Make sure to uncomment the embedding section:

```python
import pandas as pd
import numpy as np
import requests
import faiss

df = pd.read_csv('netflix_titles.csv')

def create_textual_representation(row):
    return "Type: {},\nTitle: {},\nDirector: {}\nCast: {}\nReleased: {}\nGenres: {}\n\nDescription: {}".format(
        row['type'], row['title'], row['director'], row['cast'], row['release_year'], row['listed_in'], row['description']
    )

df['textual_representation'] = df.apply(create_textual_representation, axis=1)

dim = 3072
index = faiss.IndexFlatL2(dim)
X = np.zeros((len(df), dim), dtype='float32')

for i, representation in enumerate(df['textual_representation']):
    if i % 200 == 0:
        print('Processed', i, 'instances')

    res = requests.post('http://localhost:11434/api/embeddings', json={
        'model': 'llama3',
        'prompt': representation
    })

    embedding = res.json()['embedding']
    X[i] = np.array(embedding)

index.add(X)
```

---

## Step 4: Run a Query

```python
query = "mystery movie in space"
res = requests.post('http://localhost:11434/api/embeddings', json={
    'model': 'llama3',
    'prompt': query
})
query_vec = np.array(res.json()['embedding'], dtype='float32').reshape(1, -1)
D, I = index.search(query_vec, k=5)

for idx in I[0]:
    print(df.iloc[idx]['title'], "-", df.iloc[idx]['description'])
```

---

▶️ Reference Video
Watch the complete walkthrough here: 🔗 [YouTube Project Guide](https://www.youtube.com/watch?si=COpqRri8MJgxsLxc&v=epidA1fBFtI&feature=youtu.be)

---

## Author

**Jeny Sherchan**  
MSIT Candidate, UMass Boston  
Connect via LinkedIn or email for questions!

---

## License

This project is for educational and personal use. All rights reserved to original dataset providers and authors of Ollama & FAISS.
