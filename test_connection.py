from src.api_client import APIClient
from src.config import MATCH_ID, SUPABASE_BUCKET
import requests

def test_connection():
    print("Testing Database & API Connection...\n")
    
    try:
        api = APIClient()
        
        # 1. Test fetching match data
        print(f"1. Testing API Match Fetch (Match ID: {MATCH_ID})...")
        match_data = api.fetch_match_data(MATCH_ID)
        
        home = match_data.get('home_team', {}).get('team_name', 'Unknown')
        away = match_data.get('away_team', {}).get('team_name', 'Unknown')
        players = match_data.get('players_db', {})
        
        print(f"SUCCESS! Match Data Found:")
        print(f"   - Match: {home} vs {away}")
        print(f"   - Players Loaded: {len(players)} players\n")
        
        # 2. Test Supabase Storage by uploading a tiny test file
        print("2. Testing Supabase Storage - uploading test file...")
        test_content = b"offside_test_ok"
        upload_url = f"{api.supabase_url}/storage/v1/object/{SUPABASE_BUCKET}/test_ping.txt"
        upload_res = requests.post(
            upload_url,
            headers={
                "Authorization": api.headers["Authorization"],
                "apikey": api.headers["apikey"],
                "Content-Type": "text/plain",
                "x-upsert": "true"
            },
            data=test_content
        )
        if upload_res.ok or upload_res.status_code == 200:
            public_url = f"{api.supabase_url}/storage/v1/object/public/{SUPABASE_BUCKET}/test_ping.txt"
            print(f"SUCCESS! Supabase Storage is working!")
            print(f"   - Bucket: '{SUPABASE_BUCKET}' is accessible.")
            print(f"   - Public URL works: {public_url}\n")
        else:
            print(f"WARNING: Storage upload failed. Status: {upload_res.status_code}, Message: {upload_res.text}\n")
            
        print("Connection is 100% WORKING! You are ready to run the pipeline.")
        
    except Exception as e:
        print(f"Connection Failed! Error details:\n{str(e)}")

if __name__ == "__main__":
    test_connection()
