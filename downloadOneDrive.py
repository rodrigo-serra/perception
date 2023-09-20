#!/usr/bin/env python3

from datetime import datetime
import json, os
import webbrowser, requests, msal

GRAPH_API_ENDPOINT = 'https://graph.microsoft.com/v1.0'
APP_ID = 'b72234f1-2a53-48ce-9468-1b970fd282d0'
SCOPES = ['Files.Read']
save_location = os.getcwd()

def generate_access_token(app_id, scopes):
    # Save Session Token as a token file
    access_token_cache = msal.SerializableTokenCache()

    # read the token file
    if os.path.exists('ms_graph_api_token.json'):
        access_token_cache.deserialize(open("ms_graph_api_token.json", "r").read())
        token_detail = json.load(open('ms_graph_api_token.json',))
        token_detail_key = list(token_detail['AccessToken'].keys())[0]
        token_expiration = datetime.fromtimestamp(int(token_detail['AccessToken'][token_detail_key]['expires_on']))
        if datetime.now() > token_expiration:
            os.remove('ms_graph_api_token.json')
            access_token_cache = msal.SerializableTokenCache()

    # assign a SerializableTokenCache object to the client instance
    client = msal.PublicClientApplication(client_id=app_id, token_cache=access_token_cache)
    accounts = client.get_accounts()
    
    if accounts:
        # load the session
        token_response = client.acquire_token_silent(scopes, accounts[0])
    else:
        # authetnicate your accoutn as usual
        flow = client.initiate_device_flow(scopes=scopes)
        print('user_code: ' + flow['user_code'])
        webbrowser.open('https://microsoft.com/devicelogin')
        token_response = client.acquire_token_by_device_flow(flow)

    with open('ms_graph_api_token.json', 'w') as _f:
        _f.write(access_token_cache.serialize())

    return token_response




access_token = generate_access_token(APP_ID, scopes=SCOPES)
headers = {
    'Authorization': 'Bearer' + access_token['access_token']
}


path_to_file = 'socrob/perception/detectron/models/bags/'
file_name = 'trainingInfo.txt'

# Step 2. downloading OneDrive file
response_file_content = requests.get(GRAPH_API_ENDPOINT + '/me/drive/root:/' + path_to_file + file_name +':/content', headers=headers)
with open(os.path.join(save_location, file_name), 'wb') as _f:
    _f.write(response_file_content.content)