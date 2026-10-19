import requests

try:
    response = requests.get(
        "https://huggingface.co",
        timeout=15
    )
    print("Status:", response.status_code)
except Exception as e:
    print("Error:", repr(e))