
def get_gguf(url, destination_path):
    import requests
    from tqdm import tqdm

    try:
        response = requests.get(url, stream=True)
        
        # 检查请求是否成功
        if response.status_code == 200:
            total_size = int(response.headers.get('Content-Length', 0))

            with open(destination_path, "wb") as file:
                for chunk in tqdm(response.iter_content(chunk_size=1024), 
                                  total=total_size // 1024, 
                                  unit='KB', 
                                  desc="下载中..."):
                    file.write(chunk)
            print(f"文件下载成功！保存为 {destination_path}")
        else:
            print(f"请求失败，状态码：{response.status_code}")
    except Exception as e:
        print(f"下载过程中出现错误: {e}")

'''
import requests
from tqdm import tqdm

url = "https://hf-mirror.com/mradermacher/Mistral-9B-Instruct-GGUF/resolve/main/Mistral-9B-Instruct.Q5_K_M.gguf?download=true"

response = requests.get(url, stream=True)

if response.status_code == 200:
    total_size = int(response.headers.get('Content-Length', 0))


    with open("Mistral-9B-Instruct.Q5_K_M.gguf", "wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024), total=total_size // 1024, unit='KB'):
            file.write(chunk)
    print("文件下载成功！")
else:
    print(f"请求失败：{response.status_code}")
'''