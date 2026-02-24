import requests

API_URL = "http://127.0.0.1:8000/auth/token-login"

token = st.text_input("Enter Token")

if st.button("Login"):
    response = requests.post(API_URL, params={"token": token})

    if response.status_code == 200:
        st.success("Login Successful")
        st.json(response.json())
    else:
        st.error("Invalid Token")