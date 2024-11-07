from googleapiclient.discovery import build
import os
os.environ["http_proxy"] = "http://127.0.0.1:1081"
os.environ["https_proxy"] = "http://127.0.0.1:1081"
        
def search_google(query, num_results, api_key, cse_id):
    service = build("customsearch", "v1", developerKey=api_key)
    total_results = 0
    page_size = 10
    start_index = 1
    google_result = dict()  

    while total_results < num_results:
        result = service.cse().list(q=query, cx=cse_id, num=page_size, start=start_index, gl="zh-CN").execute()
        items = result.get('items', [])
        
        for item in items:
            title = item.get('title')
            snippet = item.get('snippet')
            google_result.setdefault(title, snippet)
            total_results += 1
            if total_results >= num_results:
                break
        
        start_index += page_size
        if 'queries' not in result or 'nextPage' not in result['queries']:
            break
    
    return google_result
    

