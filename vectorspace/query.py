def vector_query(query_vector, query=None, field='_vector'):
    if query is None:
        query = {"match_all": {}}
    bool_query = {
        "bool": {
            "must": query,
            "filter": {
                "exists": {"field": field}
            }
        }
    }
    return {
        "script_score": {
            "query": bool_query,
            "script": {
                "source": f"cosineSimilarity(params.query_vector, '{field}') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
