import os
import requests

# Make sure the static folder exists
os.makedirs("static", exist_ok=True)

img_url = "https://dummyimage.com/300x450/cccccc/000000&text=No+Image"

img_data = requests.get(img_url).content

with open("static/placeholder.jpg", "wb") as f:
    f.write(img_data)

print("✅ Placeholder image saved to static/placeholder.jpg")
