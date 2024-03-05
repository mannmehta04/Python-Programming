import requests

# The URL of your Flask server
url = "http://localhost:5000/predict"  # Update with your server URL

# Sample data for the POST request
data = {
    'p_cores': 4,
    'e_cores': 8,
    'base_speed': 3.0
}

# Send a POST request to the server
response = requests.post(url, data=data)

# Check the response
if response.status_code == 200:
    print("Request successful!")
    print("Server Response:")
    print(response.json())  # If the server returns JSON data
else:
    print(f"Request failed with status code {response.status_code}")
    print("Server Response:")
    print(response.text)  # Print the response content for debugging
