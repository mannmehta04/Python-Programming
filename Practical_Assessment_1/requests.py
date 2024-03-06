import requests

url = "http://localhost:5000/predict"

data = {
    'p_cores': 4,
    'e_cores': 8,
    'base_speed': 3.0
}

response = requests.post(url, data=data)

if response.status_code == 200:
    print("Request successful!")
    print("Server Response:")
    print(response.json())
else:
    print(f"Request failed with status code {response.status_code}")
    print("Server Response:")
    print(response.text)