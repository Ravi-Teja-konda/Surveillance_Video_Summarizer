import gradio as gr
import sqlite3
import pandas as pd
import traceback

import openai
import datetime
import requests


from dotenv import load_dotenv
import os

load_dotenv()  # This will load the .env file

# Now you can securely access your environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

db_path = '/teamspace/studios/this_studio/Florence_2_video_analytics/Florence_2_video_analytics.db'


from openai import OpenAI
client = OpenAI()



def analyze_surveillance_logs(descriptions):
    '''
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": {
                    "type": "text",
                    "text": "You are a surveillance expert tasked with analyzing surveillance logs provided for each video frame. Your objective is to identify and summarize any unusual activities or noteworthy highlights based on the descriptions given for each frame. Highlight any unusual activities or issues, maintaining continuity where the context is relevant. \nThe summarization style should focus on identifying specific incidents like potential police activities, accidents, or unusual gatherings, and highlight ordinary events to provide context about the environment."
                }
            },
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": descriptions
                }
            },
            {
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": "Here's a summary of the unusual activities and highlights from the surveillance logs:\n\n**Unusual Activities:**\n- **Frame 165:** A group of people standing outside a car, one being held by another in a tense scene possibly involving police.\n- **Frame 168:** A police officer observing a parked car closely, suggesting an ongoing investigation.\n- **Frame 174:** A car accident on a residential street with multiple people involved, indicating an event requiring emergency services.\n\n**Highlights:**\n- **Recurring items:** Frequent appearance of potted plants and handbags, suggesting routine daily life.\n- **Social interactions:** Multiple frames show casual gatherings, indicating social events or breaks.\n- **Vehicular movement:** Variety of vehicles present, reflecting a typical urban environment.\n\n**Progress:** The surveillance logs capture a variety of activities, with some frames suggesting the need for further investigation or review to ensure public safety."
                }
            }
        ],
        temperature=1,
        max_tokens=128000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    '''
    response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content":"You are a surveillance expert tasked with analyzing surveillance logs provided for each video frame. Your objective is to identify and summarize any unusual activities or noteworthy highlights based on the descriptions given for each frame. Highlight any unusual activities or issues, maintaining continuity where the context is relevant. \nThe summarization style should focus on identifying specific incidents like potential police activities, accidents, or unusual gatherings, and highlight ordinary events to provide context about the environment."},
    {"role": "user", "content": descriptions},
    {"role": "assistant", "content": "Here's a summary of the unusual activities and highlights from the surveillance logs:\n\n**Unusual Activities:**\n- **Frame 165:** A group of people standing outside a car, one being held by another in a tense scene possibly involving police.\n- **Frame 168:** A police officer observing a parked car closely, suggesting an ongoing investigation.\n- **Frame 174:** A car accident on a residential street with multiple people involved, indicating an event requiring emergency services.\n\n**Highlights:**\n- **Recurring items:** Frequent appearance of potted plants and handbags, suggesting routine daily life.\n- **Social interactions:** Multiple frames show casual gatherings, indicating social events or breaks.\n- **Vehicular movement:** Variety of vehicles present, reflecting a typical urban environment.\n\n**Progress:** The surveillance logs capture a variety of activities, with some frames suggesting the need for further investigation or review to ensure public safety."}

  ],
  temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
  )
    
    if response is not None:
    # Accessing the first choice's message content
      content = response.choices[0].message.content
    return content


def summarize_logs(query_prompt, start_datetime, end_datetime):
    conn = sqlite3.connect(db_path)
    
    # Correcting the format to match the input strings
    format_string = '%Y-%m-%d %I:%M:%S %p'  # Updated format string to match the input
    print(f"Received start datetime: {start_datetime}")
    print(f"Received end datetime: {end_datetime}")

    try:
        start_dt = pd.to_datetime(start_datetime, format=format_string)
        end_dt = pd.to_datetime(end_datetime, format=format_string)
    except ValueError as e:
        print(f"Error parsing datetime strings: {e}")
        return "Invalid datetime format. Please use 'YYYY-MM-DD HH:MM:SS AM/PM' format."

    # Format the datetime for SQL query
    start_dt_formatted = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_dt_formatted = end_dt.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Formatted start datetime for SQL: {start_dt_formatted}")
    print(f"Formatted end datetime for SQL: {end_dt_formatted}")

    query = """
        SELECT * FROM frame_data
        WHERE timestamp BETWEEN ? AND ?
    """
    df = pd.read_sql_query(query, conn, params=(start_dt_formatted, end_dt_formatted))
    conn.close()
    
    if df.empty:
        return "No data available for the specified timeframe."
    else:
        descriptions = ' '.join(df['result'].astype(str).tolist())
        if descriptions: 
          descriptions = analyze_surveillance_logs(descriptions)
          return descriptions


def launch_interface():
    interface = gr.Interface(
        fn=summarize_logs,
        inputs=[
            gr.Textbox(label="Enter your query about the surveillance logs (e.g., unusual activities, highlights)", placeholder="Type here...", value=""),
            gr.Textbox(label="Start DateTime (format: YYYY-MM-DD HH:MM:SS AM/PM)", placeholder="2024-05-25 06:20:00 PM", value="2024-05-25 06:20:00 PM"),
            gr.Textbox(label="End DateTime (format: YYYY-MM-DD HH:MM:SS AM/PM)", placeholder="2024-05-25 06:21:00 PM", value="2024-05-25 06:21:00 PM")
        ],
         outputs=gr.Textbox(label="Output", lines=20,max_lines=10000, placeholder="Results will appear here..."),  # `lines=10` ensures that the box is initially sized to show 10 lines at a time # Adjust height as needed
        title="Surveillance Video Summarizer",
        description="Enter your query and the time range to analyze surveillance logs."
    )
    interface.launch(share=True)


if __name__ == "__main__":
   launch_interface()

