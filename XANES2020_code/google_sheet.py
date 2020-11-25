import pickle
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from datetime import datetime
import pandas as pd
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SAMPLE_SPREADSHEET_ID = '1jrTqDuzvn_oGjfZC0Klntps_1PgscXvMRw93QErpiQc'
# The ID and range of a sample spreadsheet.


class google_sheet_retrieval:
    
    
    
    def __init__(self):
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        self.service = build('sheets', 'v4', credentials=creds)
        self.get_sheet_dataframe()
        self.fill_in_blanks()

    def get_sheet_dataframe(self,sample_range = 'ShotSheet!A1:AT'):
        sheet = self.service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                            range=sample_range).execute()
        values = result.get('values', [])
        self.headings = values[0]
        self.sheet_df = pd.DataFrame(values[1:],columns=self.headings)
        return self.sheet_df

    def fill_in_blanks(self):
        prev_row = None
        for n,row in self.sheet_df.iterrows():
            if prev_row is None:
                prev_row = row
            else:
                for key in self.headings:
                    if str(row[key]) =='' or row[key] is None:
                        row[key] = prev_row[key]
                    else:
                        prev_row[key] = row[key]

    def get_run(self,run_name):

        run_dt = datetime.strptime(run_name[:8],'%Y%m%d')
        run_date = run_dt.strftime('%d/')+run_dt.strftime('%m/').lstrip('0')+run_dt.strftime('%Y')
        run_run = int(run_name[12:])
        match_df = self.sheet_df.loc[self.sheet_df['date'].str.match(run_date)]
        match_df = match_df.loc[self.sheet_df['run'].str.match(str(run_run))]
        if len(match_df)<1:
            run_date = run_dt.strftime('%d/')+run_dt.strftime('%m/').lstrip('0')+run_dt.strftime('%y')
            match_df = self.sheet_df.loc[self.sheet_df['date'].str.match(run_date)]
            match_df = match_df.loc[self.sheet_df['run'].str.match(str(run_run))]
        if len(match_df)<1:
            run_date = run_dt.strftime('%d/')+run_dt.strftime('%m/')+run_dt.strftime('%Y')
            match_df = self.sheet_df.loc[self.sheet_df['date'].str.match(run_date)]
            match_df = match_df.loc[self.sheet_df['run'].str.match(str(run_run))]
        self.match_df = match_df
        return match_df
    
    

