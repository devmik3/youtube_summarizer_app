# 1. Imports
import streamlit as st
import os
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from pytube.extract import initial_data, initial_player_response
from pytube.request import get
import logging
import json
from dotenv import load_dotenv

# 2. Configuration and initialization
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# 3. Helper functions
def fetch_video_metadata(video_ids):
    metadata = {}
    for video_id in video_ids:
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            logging.debug(f"Attempting to fetch metadata for URL: {url}")

            # Manually fetch the watch page HTML
            watch_html = get(url)
            logging.debug(f"Successfully fetched watch HTML for video {video_id}")

            # Extract the initial player response
            player_response = initial_player_response(watch_html)
            logging.debug(f"Successfully extracted initial player response for video {video_id}")

            # Extract title from player response if possible
            title = player_response.get('videoDetails', {}).get('title', 'Unknown Title')

            # Store only the title
            metadata[video_id] = {
                'title': title
            }
            logging.info(f"Successfully fetched title for video {video_id}: {title}")

        except Exception as e:
            logging.error(f"Failed to fetch metadata for video {video_id}: {str(e)}")
    return metadata

def fetch_transcripts_and_metadata(video_ids, language_codes=('en',)):
    transcripts = {}
    metadata = fetch_video_metadata(video_ids)
    for video_id in video_ids:
        try:
            logging.debug(f"Attempting to fetch transcript for video {video_id}")
            # Fetch transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=language_codes)
            full_transcript = ' '.join([item['text'] for item in transcript_list])
            transcripts[video_id] = full_transcript
            logging.info(f"Successfully fetched transcript for video {video_id}")
        except Exception as e:
            logging.error(f"Failed to fetch transcript for video {video_id}: {str(e)}")
    return transcripts, metadata

def summarize_transcripts(transcripts, metadata):
    summaries = {}
    for video_id, transcript in transcripts.items():
        try:
            title = metadata.get(video_id, {}).get('title', '')

            # Construct the context by combining title and transcript
            context = f"**Title:** {title}\n\n**Transcript:** {transcript}"

            # Define the prompt with context
            prompt = f"{context}\n\nSummarize the above content with attention to correcting any transcription errors based on the title provided."

            # Use the OpenAI client to create a chat completion
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # Use the correct model name
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": """
AI Summarization Assistant Instructions
Objective: You are tasked with summarizing transcripts of the researches. Your goal is to identify and consolidate the main ideas from each segment of the transcript, especially when the content includes overlapping topics or discussions by different people. The summary should be provided in a well-organized markdown format, focusing on clarity, conciseness, and logical flow.

Guidelines for the Summary:
    1. Identify Main Ideas:
        ◦ Extract the primary concepts, arguments, and themes from each section under #TITLE.
        ◦ Combine insights on the same topic from different segments or speakers into a single coherent idea.
    2. Hierarchical Structure:
        ◦ Section Headers: Use markdown headers to denote each major part or division of the content.
            ▪ Example: ## Main Topic
        ◦ Subheaders: Break down each section into subtopics or themes.
            ▪ Example: ### Key Subtopic
        ◦ Bulleted Lists: Use bullet points to highlight key points within each subtopic or theme.
            ▪ Example:
                • - Key Point 1
                • - Key Point 2
    3. Detailed Content:
        ◦ Subtopic Names: Each subheader should begin with the subtopic or theme name.
        ◦ Key Points: Provide detailed bulleted points that cover the essence of the main ideas, arguments, and themes presented.
            ▪ Be specific and informative.
            ▪ Capture the essence without unnecessary detail.
    4. Clarity and Conciseness:
        ◦ Use clear, straightforward language to convey the content.
        ◦ Avoid jargon and overly complex phrasing unless essential to the understanding of the topic.
        ◦ Each point should be concise yet rich in information.
    5. Logical Flow:
        ◦ Maintain a coherent and logical progression throughout the summary.
        ◦ Ensure smooth transitions between sections and subtopics to facilitate easy reading and understanding.

Instructions for Response:
    • Focus solely on delivering an informative and structured summary.
    • Do not repeat these instructions within the summary.
    • Avoid making apologies or self-referencing statements.
    • Present the final output in markdown format for easy reading and further editing.

Example of the Expected Output Format:
## [Main Topic 1]: Overview of the Discussion

### [Key Subtopic 1]
- **[Main Idea 1]**: Description of the key point.
- **[Main Idea 2]**: Description of another key point.

### [Key Subtopic 2]
- **[Key Point 1]**: Explanation of the primary argument.
- **[Key Point 2]**: Explanation of supporting evidence or arguments.

## [Main Topic 2]: Further Insights

### [Key Subtopic 1]
- **[Main Idea 1]**: Summary of the main discussion points.
- **[Main Idea 2]**: Further details on supporting arguments or examples.

Steps to Start the Summarization:
    1. Analyze the transcript content.
    2. The assistant will extract and summarize the key ideas from each section as outlined above.
    3. Once the summary is provided, you can ask specific questions or request more details on topics of interest.
                        """
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            # Access the message content correctly
            summary = completion.choices[0].message.content.strip()
            summaries[video_id] = summary
            logging.info(f"Successfully summarized transcript for video {video_id}")
        except Exception as e:
            logging.error(f"Failed to summarize transcript for video {video_id}: {str(e)}")
    return summaries

def check_and_improve_summary(summaries):
    improved = {}
    for video_id, summary in summaries.items():
        try:
            # Improvement prompt
            prompt = f"""
The following is a summary generated from a research. Please use your general knowledge to correct any spelling or grammatical errors, improve the clarity of the summary, and ensure it accurately reflects the main points. Additionally, please perform the following tasks:

1. Create a TL;DR at the beginning of the summary that captures the essence of the research in one or two sentences.
2. Remove any content or references that are not directly related to the main topic. This includes but is not limited to:
   - Encouragement to subscribe, share, or like the video.
   - Requests for comments, feedback, or sharing of insights.
   - Any mentions of promoting the YouTube channel itself.
   - Any phrases like 'The speaker discusses', 'The video emphasizes', 'The video examines', 'The presenter uses', etc.
3. Ensure that the conclusion is focused solely on summarizing the content and insights of the research, not on promoting the YouTube channel or encouraging viewer interaction.

Summary:
{summary}

Provide the improved summary below:
"""

            # Use the OpenAI client to create a chat completion
            completion = client.chat.completions.create(
                model="chatgpt-4o-latest",  # Use the correct model name
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that checks and improves summaries."
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            improved_summary = completion.choices[0].message.content.strip()
            improved[video_id] = improved_summary
            logging.info(f"Successfully improved summary for video {video_id}")
        except Exception as e:
            logging.error(f"Failed to improve summary for video {video_id}: {str(e)}")
            improved[video_id] = summary  # Fallback to original summary if improvement fails
    return improved

# 4. Main processing function
def process_videos(video_urls):
    # Extract video IDs from URLs
    video_ids = [url.split('v=')[-1] for url in video_urls if 'v=' in url]

    if not video_ids:
        st.error("No valid YouTube video IDs found in the provided URLs.")
        return

    # Fetch transcripts and metadata
    with st.spinner('Fetching transcripts and metadata...'):
        transcripts, metadata = fetch_transcripts_and_metadata(video_ids)

    # Summarize transcripts
    with st.spinner('Summarizing transcripts...'):
        summaries = summarize_transcripts(transcripts, metadata)

    # Improve summaries
    with st.spinner('Improving summaries...'):
        improved_summaries = check_and_improve_summary(summaries)

    # Display summaries
    st.header("Summaries")
    
    # Option to save summaries
    if st.button("Save Summaries"):
        # Save to markdown file
        with open('summaries.md', 'w', encoding='utf-8') as f:
            for video_id, summary in improved_summaries.items():
                title = metadata[video_id].get('title', 'Unknown Title')
                url = f"https://www.youtube.com/watch?v={video_id}"
                f.write(f"## Summary for Video: {title}\n\n{url}\n\n{summary}\n\n")
        st.success("Summaries saved to 'summaries.md'")

        # Optionally save transcripts to JSON
        # with open('summaries.json', 'w', encoding='utf-8') as f:
        #     json.dump(transcripts, f, ensure_ascii=False, indent=4)
        # st.success("Transcripts saved to 'summaries.json'")

    # Display summaries in the Streamlit interface
    for video_id, summary in improved_summaries.items():
        title = metadata[video_id].get('title', 'Unknown Title')
        url = f"https://www.youtube.com/watch?v={video_id}"
        st.subheader(f"Summary for Video: {title}")
        st.markdown(f"[Link to video]({url})")
        st.markdown(summary)

# 5. Main application function
def main():
    st.title("YouTube Video Transcription and Summarization")

    st.write("Enter one or more YouTube video URLs (one per line):")
    video_urls_input = st.text_area("YouTube Video URLs")

    if st.button("Fetch and Summarize"):
        video_urls = video_urls_input.strip().split('\n')
        if video_urls:
            st.info("Processing videos...")
            process_videos(video_urls)
        else:
            st.warning("Please enter at least one YouTube video URL.")

# 6. Entry point
if __name__ == "__main__":
    main()


