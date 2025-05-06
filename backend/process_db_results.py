def process_db_select_results(query_results):
    results_to_show = []
    for query_result_row in query_results:
        results_to_show.append(
            {
                "document_name": query_result_row.document_name,
                "document_path": query_result_row.document_path,
                "snippet": query_result_row.snippet
            }
        )
    return results_to_show

