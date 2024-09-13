# 🎥 Surveillance Video Summarizer: AI-Powered Video Analysis and Summarization
*Checked on 16.09.2024 ✅*

**Surveillance Video Summarizer** is a robust AI-driven system that processes surveillance videos, extracts key frames, and generates detailed annotations. Powered by a fine-tuned Florence-2 Vision-Language Model (VLM) specifically trained on the SPHAR dataset, it highlights notable events, actions, and objects within video footage and logs them for easy review and further analysis.

⚠️ **Important**: This system is intended for surveillance professionals or developers working with surveillance data. Ensure responsible use of AI in your projects to maintain privacy and security standards.

The fine-tuned model can be found at: [kndrvitja/florence-SPHAR-finetune-2](https://huggingface.co/kndrvitja/florence-SPHAR-finetune-2).

See the tool in action below!

### 🎥 Demo Video

[![Demo Video](https://img.youtube.com/vi/37MydYtoo4U/sddefault.jpg)](https://youtu.be/37MydYtoo4U)
---

## Features

- **AI-Powered Video Summarization**  
  Automatically extract frames from surveillance videos and generate high-quality annotations that capture actions, interactions, objects, and unusual events. The annotations are stored in a SQLite database for easy retrieval.

- **Real-Time Frame Processing**  
  By utilizing asynchronous threading, the system processes video frames efficiently, allowing real-time analysis while minimizing performance bottlenecks. It logs every step, ensuring easy debugging and verification.

- **Fine-Tuned Florence-2 VLM for SPHAR Dataset**  
  The summarization process is powered by a fine-tuned Florence-2 VLM, specifically trained on the SPHAR dataset. This model is optimized to detect and describe surveillance-specific events with higher accuracy.

- **Gradio-Powered Interactive Interface**  
  Interact with the surveillance logs through a Gradio-based web interface. You can specify time ranges, and the system will retrieve the annotated logs, providing insights into the video footage over the given period.

---

## 📣 How it Works

1. **Frame Extraction**:  
   Frames are extracted at regular intervals from surveillance video files using OpenCV.
   
2. **AI-Powered Annotation**:  
   Each frame is analyzed by the fine-tuned Florence-2 Vision-Language Model, generating insightful annotations about the scene.
   
3. **Data Storage**:  
   Annotations and their associated frame data are stored in a SQLite database, ready for future analysis.
   
4. **Gradio Interface**:  
   Allows users to easily query the surveillance logs by providing a time range and specific prompts.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Surveillance_Video_Summarizer.git
