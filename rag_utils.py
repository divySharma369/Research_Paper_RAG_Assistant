import numpy as np

def truncate_context(text, max_tokens=350):
    words = text.split()
    return " ".join(words[:max_tokens])

def compute_confidence(similarity_scores):
    avg_score = float(np.mean(similarity_scores))
    return round(avg_score * 100, 2)

def retrieve_with_sources(query, embedding_model, index, documents, titles, categories, top_k=1):
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).reshape(1, -1)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "document": documents[idx],
            "title": titles[idx],
            "categories": categories[idx],
            "similarity": float(score)
        })

    return results

def rag_pipeline(query, embedding_model, index, documents, titles, categories, top_k=1):
    results = retrieve_with_sources(
        query,
        embedding_model,
        index,
        documents,
        titles,
        categories,
        top_k
    )

    context = "\n\n".join([r["document"] for r in results])
    safe_context = truncate_context(context)

    similarity_scores = [r["similarity"] for r in results]
    confidence = compute_confidence(similarity_scores)

    return safe_context, results, confidence
