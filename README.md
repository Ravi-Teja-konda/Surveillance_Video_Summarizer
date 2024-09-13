# 🎥 Surveillance Video Summarizer: AI-Powered Video Analysis and Summarization
*Checked on 13.09.2024 ✅*

**Surveillance Video Summarizer** is a AI-driven system that processes surveillance videos, extracts key frames, and generates detailed annotations. Powered by a **fine-tuned Florence-2 Vision-Language Model (VLM)** specifically trained on the SPHAR dataset, it highlights notable events, actions, and objects within video footage and logs them for easy review and further analysis.

The fine-tuned model can be found at: [kndrvitja/florence-SPHAR-finetune-2](https://huggingface.co/kndrvitja/florence-SPHAR-finetune-2).

See the tool in action below!

### 🎥 [Demo Video](https://youtu.be/37MydYtoo4U)

[![Demo Video](https://img.youtube.com/vi/37MydYtoo4U/sddefault.jpg)](https://youtu.be/37MydYtoo4U)
---

## Features

- **AI-Powered Video Summarization**  
  Automatically extract frames from surveillance videos and generate annotations that capture actions, interactions, objects, and unusual events. The annotations are stored in a SQLite database for easy retrieval.

- **Real-Time Frame Processing**  
  By utilizing asynchronous threading, the system processes video frames efficiently, allowing real-time analysis while minimizing performance bottlenecks. It logs every second, ensuring easy debugging and verification.

- **Fine-Tuned Florence-2 VLM for SPHAR Dataset**  
  The summarization process is powered by a fine-tuned Florence-2 VLM, specifically trained on the SPHAR dataset. This model is optimized to detect and describe surveillance-specific events with higher accuracy.

- **Gradio-Powered Interactive Interface**  
Interact with the surveillance logs through a Gradio-based web interface. You can specify time ranges, and the system will retrieve, summarize, and analyze the annotated logs, providing detailed insights into the video footage over the selected period using the OpenAI API. This functionality can be extended to leverage advanced models like Gemini, enabling more efficient handling of longer context videos and delivering more comprehensive video summarization over extended timeframes.

---

## 📣 How it Works

1. **Frame Extraction**:  
   Frames are extracted at regular intervals from surveillance video files using OpenCV.
   
2. **AI-Powered Annotation**:  
   Each frame is analyzed by the fine-tuned Florence-2 Vision-Language Model, generating insightful annotations about the scene.
   
3. **Data Storage**:  
   Annotations and their associated frame data are stored in a SQLite database, ready for future analysis.
   
4. **Gradio Interface**:
   The system allows users to effortlessly query surveillance logs by providing a specific time range and tailored prompts. It retrieves, summarizes, and analyzes the relevant video footage, offering concise insights

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ravi-Teja-konda/Surveillance_Video_Summarizer.git
   ```
2. **Navigate to the project directory**:
  ```bash
  cd Surveillance_Video_Summarizer
  ```
3. **Install the required Python libraries**:
```bash
pip install -r requirements.txt
```
## **Configuration**
Model and Processor

- The system utilizes the Florence-2 Vision-Language Model fine-tuned for the SPHAR dataset. The fine-tuned model can be found at kndrvitja/florence-SPHAR-finetune-2.

- Ensure you have your OpenAI API key stored in a .env file as required.

Database Path

- The default SQLite database for storing frame data is located at /teamspace/studios/Florence_2_video_analytics/Florence_2_video_analytics.db. You can modify this path.

## **Usage**
Firstly, run the frame extraction :

```bash
python surveillance_video_summarizer.py
```
Next, interact with the Gradio interface for log analysis:
```bash
python surveillance_log_analyzer_with_gradio.py
```
From here, you can use the Gradio interface to query specific periods of video footage and retrieve annotated summaries based on your input.
You can query the system for specific actions, notable events, or general activity summaries. Provide the time range and your query prompt, and the system will return the relevant logs

## 🚀 Future Enhancements

### Advanced Event Detection
We plan to enhance the model’s capability to detect more complex events such as traffic violations, suspicious behavior, and other nuanced surveillance scenarios by training florence-2 with more data

### Real-Time Streaming
In future will plan to support real-time video streams for immediate frame extraction and analysis as the video is being captured.

---

## Contributing
Contributions are welcome! Feel free to submit a pull request.

---

## ❤️ Support the Project
If you find this project useful, consider starring it on GitHub to help others discover it!

---

## 📚 References
Inspired by advances in Vision-Language models like Florence-2.

- https://arxiv.org/pdf/2311.06242
- https://huggingface.co/papers/2311.06242
- https://github.com/retkowsky/florence-2


## License
This project is licensed under the Apache License 2.0.


