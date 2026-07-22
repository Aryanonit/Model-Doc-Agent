"""
Run this ONCE, locally, to authorize DocAgent to create Google Docs as YOU
(not as the service account). This fixes the 'storageQuotaExceeded' error,
since service accounts have 0 bytes of Drive storage and can't own files,
even inside a shared folder.

Setup before running:
1. In Cloud Console -> APIs & Services -> Credentials, find your existing
   "ModelDocAgent" OAuth 2.0 Client (Type: Desktop).
2. Click it -> Download JSON -> save it in this same folder as client_secret.json
3. pip install google-auth-oauthlib
4. python authorize_google.py
   -> This opens a browser, asks you to log in as awaraaalu@gmail.com and
      approve access. After approving, it saves token.json here.
5. Never commit client_secret.json or token.json to git -- add both to .gitignore.
"""
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/drive.file'
]

def main():
    flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
    creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as f:
        f.write(creds.to_json())
    print("✅ Authorization complete. Saved token.json.")
    print("   DocAgent will now create Docs owned by your account, not the service account.")

if __name__ == '__main__':
    main()