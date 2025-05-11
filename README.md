
# Sentiment Analysis on Polling Response Audios

This project performs sentiment analysis on polling response audios by transcribing the audio files and analyzing the resulting text for sentiment. The goal is to extract meaningful insights from public opinion data across various topics such as politics, healthcare, economy, and social issues.

## Key Features

- **Audio Transcription**: Transcribes audio polling responses into text for further analysis.
- **Multi-Model Sentiment Analysis**: Utilizes several sentiment analysis models, including:
  - Twitter-roberta
  - DeBERTa Sentiment
  - DistilBERT Emotion Model
  - TextBlob (for polarity-based sentiment classification)
- **Topic Detection**: Analyzes responses for keywords related to key topics such as Immigration, Healthcare, Economy, Race Relations, and more.
- **Visualization**: Generates visualizations to display sentiment distribution and topic-based sentiment analysis.
- **Statistical Analysis**: Uses repeated measures ANOVA and one-way ANOVA to evaluate the effectiveness and consistency of different sentiment models.

## Setup

### Prerequisites

- Python 3.x
- Required Libraries:
  - `transformers`
  - `textblob`
  - `tqdm`
  - `matplotlib`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `openpyxl` (for Excel output)
  
Install dependencies:

```bash
pip install transformers textblob tqdm matplotlib scikit-learn pandas numpy openpyxl
```

### Project Structure

```
sentiment-analysis-project/
│
├── input_data/             # Directory containing the audio-to-text transcriptions
├── models/                 # Directory containing pre-trained models
│   ├── twitter_roberta/
│   ├── deberta_sentiment/
│   ├── distilbert_emotion/
│
├── results/                # Directory where results and visualizations are saved
│   ├── audio_sentiment_results.csv
│   ├── audio_sentiment_summary.xlsx
│   ├── overall_sent_dist.png
│   ├── sentiment_by_topic.png
│
├── script.py               # Main Python script for processing
├── README.md               # This README file
```

## Usage

1. **Prepare Your Data**: Place your speech-to-text transcription files in the `input_data/` directory.
2. **Run the Script**: Execute the main script to process the data and generate sentiment analysis results.

```bash
python script.py
```

3. **Output**: The analysis results will be saved as:
   - `audio_sentiment_results.csv`: A CSV file containing sentiment labels and topic-specific keyword analysis.
   - `audio_sentiment_summary.xlsx`: An Excel summary with detailed results.
   - Visualizations in the `results/` directory:
     - Sentiment distribution charts.
     - Topic-based sentiment analysis charts.

## Models Used

- **Twitter-roberta**: Fine-tuned RoBERTa model trained for sentiment classification in short-form text.
- **DeBERTa Sentiment**: DeBERTa-based model optimized for sentiment analysis.
- **DistilBERT Emotion**: DistilBERT model fine-tuned for emotion detection.
- **TextBlob**: A simple NLP library for polarity-based sentiment classification.

## Visualizations

1. **Confusion Matrices**: Shows the agreement between different sentiment models (e.g., Twitter vs. DeBERTa).
2. **Overall Sentiment Distribution**: A bar chart displaying the overall sentiment distribution across all responses.
3. **Sentiment by Topic**: A bar chart visualizing sentiment classification by different topics (e.g., Healthcare, Economy).
4. **Emotion Distribution**: A bar chart displaying emotion distribution (happy, sad, angry, etc.).

## Statistical Analysis

- **ANOVA Analysis**: Repeated measures ANOVA and one-way ANOVA are applied to assess the consistency and effectiveness of sentiment predictions across multiple models.

## Conclusion

This project leverages state-of-the-art sentiment analysis models to provide actionable insights into public opinion from polling responses. The results can help understand sentiment trends across various issues and improve decision-making based on public perception.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
