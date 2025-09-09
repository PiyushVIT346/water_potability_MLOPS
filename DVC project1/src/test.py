#https://water-test-1-a6qt.onrender.com/predict
import json
import requests

url="https://water-test-1-a6qt.onrender.com/predict"

x_new=dict(
    ph=10.71608,
    Hardness=215.3087,
    Solids=157.502,
    Chlorides=13.298,
    Sulfate=3.543,
    Conductivity=189.75,
    Organic_carbon=15.3,
    Trihalomethanes=0.0,
    Turbidity=4.5,
    Chloramines=2.5,
)

x_new_json=json.dumps(x_new)

response=requests.post(url,data=x_new_json)

print("Response Text: " + response.text)
print("Response Status Code: " + str(response.status_code))