from docs_manager.vector_store import search_documents


results = search_documents(library="pandas", query="dataframe", k=10)
print(results)
