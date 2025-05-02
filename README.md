# Nlp-Document-Scanner


An application that processes multiple text documents at once, performing various NLP analyses including summarization, key phrase extraction, named entity recognition, and topic modeling.
![image](https://github.com/user-attachments/assets/37ae3874-765c-4eb1-b80c-1b7a7798e0ff)

## Features

- **Batch Processing**: Process multiple documents at once
- **Document Summarization**: Generate extractive summaries of variable length
- **Key Phrase Extraction**: Identify important terms and phrases
- **Named Entity Recognition**: Extract and categorize entities like organizations, locations, and people
- **Topic Modeling**: Discover underlying topics in your documents
- **Export Results**: Save analysis results to CSV for further processing

## Installation

1. Ensure you have Python 3.7+ installed on your system
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the required spaCy model (will be done automatically on first run if not present):

```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Run the application:

```bash
python nlp_document_scanner.py
```

2. Add text files (.txt) to process using the "Add Files" button
3. Select the analyses you want to perform
4. Click "Process Documents" to start the analysis
5. View results in the "Results" tab
6. Export results to CSV if needed

## Technical Details

### Text Processing Pipeline

1. **Text Cleaning**: Lowercasing, stopword removal, lemmatization
2. **Key Phrase Extraction**: Uses TF-IDF to identify important terms
3. **Summarization**: Extracts key sentences based on importance scores
4. **Named Entity Recognition**: Uses spaCy to identify and categorize entities
5. **Topic Modeling**: Non-negative Matrix Factorization (NMF) for topic discovery

### Libraries Used

- **NLTK**: For tokenization, stopword removal, and lemmatization
- **spaCy**: For named entity recognition
- **scikit-learn**: For TF-IDF vectorization and NMF topic modeling
- **Pandas**: For data manipulation and CSV export
- **Tkinter**: For the graphical user interface

## Customization

- Adjust summary length by changing the "Summary Length" value
- Enable/disable specific analyses using the checkboxes
- Modify the code to add additional NLP processing techniques

## Troubleshooting

- If you encounter errors related to missing NLTK resources, run:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('averaged_perceptron_tagger')
  ```

- If spaCy model loading fails, manually install it:
  ```bash
  python -m spacy download en_core_web_sm
  ```
