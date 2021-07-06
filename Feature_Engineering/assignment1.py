import requests
from dotenv import dotenv_values
config = dotenv_values("../.env")


def generate_url(hash):
    api_key = config["VI_API"]
    headers = {
        "Content-type": "application/json",
        "X-Apikey": api_key
    }
    url = "https://www.virustotal.com/api/v3/files/{}".format(hash)
    return url, headers


def get_malware_info(file_hash):
    url, headers = generate_url(file_hash)
    malware_info_response = requests.get(url, headers=headers)
    res_malware_json = malware_info_response.json()
    return res_malware_json


if __name__ == '__main__':
    malware_hashes = ["0a0cf5c7ef7cfd16e14c5c5ede602528", "0a0f706955b59cbda99e12a345d8e8c6", "0a1a6972c80e720917f70fdac849776b", "0a1a11341a6515c791eb700f8f709dc3", "0a1b2ec06f7b263cf687a393f43b18c0", "0a1b7777b3e014ad64db66090997046e", "0a1bdb29862fb7c3ffdd139d332f499d", "0a1cfd19202f56bf440329e7e670647a", "0a1dbd7853c9cd817d355893ba012e00", "0a1eaea5721445fc70b461c9d64fdda2"]
    for i in range(len(malware_hashes)):
        malware_hash = malware_hashes[i]
        print(get_malware_info(malware_hash))
