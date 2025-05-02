import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import spacy
import threading
import queue
import time

# Download necessary NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

# Load spaCy model
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

class TextProcessor:
    def __init__(self):
        download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = None  # Will be loaded when needed
    
    def load_spacy(self):
        if self.nlp is None:
            self.nlp = load_spacy_model()
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        words = word_tokenize(text)
        # Remove stopwords and non-alphabetic characters
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        # Lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    
    def extract_key_phrases(self, text, top_n=10):
        """Extract key phrases using TF-IDF"""
        sentences = sent_tokenize(text)
        
        # Skip if there are too few sentences
        if len(sentences) < 3:
            return ["Not enough text for meaningful extraction"]
        
        # Clean sentences
        cleaned_sentences = [self.clean_text(sentence) for sentence in sentences]
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=200)
        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
            
            # If matrix is empty, return early
            if tfidf_matrix.shape[1] == 0:
                return ["Not enough repeating terms for extraction"]
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top words based on TF-IDF scores
            tfidf_scores = zip(feature_names, np.asarray(tfidf_matrix.sum(axis=0)).ravel())
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
            
            # Return top phrases
            return [word for word, score in sorted_scores[:top_n]]
        except:
            return ["Not enough data for meaningful extraction"]
    
    def generate_summary(self, text, num_sentences=3):
        """Generate extractive summary"""
        sentences = sent_tokenize(text)
        
        # Return early if there are fewer sentences than requested
        if len(sentences) <= num_sentences:
            return text
        
        # Clean sentences
        cleaned_sentences = [self.clean_text(sentence) for sentence in sentences]
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
            
            # Calculate sentence scores
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            
            # Get indices of top sentences
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            # Extract and join the top sentences
            summary = ' '.join([sentences[i] for i in top_indices])
            return summary
        except:
            # If vectorization fails, return the first few sentences
            return ' '.join(sentences[:num_sentences])
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        self.load_spacy()
        doc = self.nlp(text[:10000])  # Limit text size for performance
        
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def extract_topics(self, text, num_topics=3, num_words=5):
        """Extract topics using NMF"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Create documents (split by paragraphs)
        documents = [p for p in cleaned_text.split('\n\n') if len(p) > 50]
        
        # If there are too few documents, split by sentences
        if len(documents) < num_topics:
            documents = [s for s in sent_tokenize(cleaned_text) if len(s) > 30]
        
        # Return early if still not enough data
        if len(documents) < num_topics:
            return [["Insufficient data for topic modeling"]]
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        try:
            tfidf = vectorizer.fit_transform(documents)
            
            # NMF Topic Modeling
            nmf_model = NMF(n_components=num_topics, random_state=42)
            nmf_model.fit(tfidf)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get topics
            topics = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[:-num_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(top_words)
            
            return topics
        except:
            return [["Insufficient data for topic modeling"]]

class DocumentScanner:
    def __init__(self, root):
        self.root = root
        self.root.title("NLP Document Scanner")
        self.root.geometry("1000x700")
        
        self.processor = TextProcessor()
        self.files = []
        self.results = []
        self.progress_queue = queue.Queue()
        
        self.create_widgets()
        self.update_progress()
    
    def create_widgets(self):
        # Create frames
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Input tab
        input_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(input_frame, text="Input")
        
        # File selection widgets
        ttk.Label(input_frame, text="Selected Files:").pack(anchor=tk.W)
        self.file_listbox = tk.Listbox(input_frame, width=80, height=10)
        self.file_listbox.pack(fill=tk.X, pady=5)
        
        file_buttons_frame = ttk.Frame(input_frame)
        file_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_buttons_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_buttons_frame, text="Clear Files", command=self.clear_files).pack(side=tk.LEFT)
        
        # Options frame
        options_frame = ttk.LabelFrame(input_frame, text="Analysis Options", padding=10)
        options_frame.pack(fill=tk.X, pady=10)
        
        # Checkbuttons for analysis options
        self.option_vars = {
            "extract_summary": tk.BooleanVar(value=True),
            "extract_key_phrases": tk.BooleanVar(value=True),
            "extract_entities": tk.BooleanVar(value=True),
            "extract_topics": tk.BooleanVar(value=True)
        }
        
        ttk.Checkbutton(options_frame, text="Generate Summary", variable=self.option_vars["extract_summary"]).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Extract Key Phrases", variable=self.option_vars["extract_key_phrases"]).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Extract Named Entities", variable=self.option_vars["extract_entities"]).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Extract Topics", variable=self.option_vars["extract_topics"]).grid(row=1, column=1, sticky=tk.W)
        
        # Summary length
        summary_frame = ttk.Frame(options_frame)
        summary_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(summary_frame, text="Summary Length (sentences):").pack(side=tk.LEFT)
        self.summary_length = tk.IntVar(value=3)
        ttk.Spinbox(summary_frame, from_=1, to=10, width=5, textvariable=self.summary_length).pack(side=tk.LEFT, padx=5)
        
        # Process button
        self.process_button = ttk.Button(input_frame, text="Process Documents", command=self.process_documents)
        self.process_button.pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(input_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.progress_label = ttk.Label(input_frame, text="")
        self.progress_label.pack(anchor=tk.W)
        
        # Results tab (will be populated after processing)
        self.results_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.results_frame, text="Results")
        
        # Export button
        export_frame = ttk.Frame(self.results_frame)
        export_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        ttk.Button(export_frame, text="Export to CSV", command=self.export_results).pack(side=tk.RIGHT)
    
    def update_progress(self):
        """Update progress bar and label from queue"""
        try:
            while True:
                progress, message = self.progress_queue.get_nowait()
                self.progress_var.set(progress)
                self.progress_label.config(text=message)
                self.root.update_idletasks()
        except queue.Empty:
            pass
        self.root.after(100, self.update_progress)
    
    def add_files(self):
        """Add files to the list"""
        new_files = filedialog.askopenfilenames(
            title="Select files",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        
        for file in new_files:
            if file not in self.files:
                self.files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))
    
    def clear_files(self):
        """Clear the file list"""
        self.files = []
        self.file_listbox.delete(0, tk.END)
    
    def process_documents(self):
        """Process all selected documents"""
        if not self.files:
            tk.messagebox.showwarning("No Files", "Please add files to process")
            return
        
        # Disable the process button during processing
        self.process_button.config(state=tk.DISABLED)
        self.results = []
        
        # Start processing in a separate thread
        threading.Thread(target=self._process_documents_thread, daemon=True).start()
    
    def _process_documents_thread(self):
        """Thread function for document processing"""
        try:
            total_files = len(self.files)
            
            for i, file_path in enumerate(self.files):
                file_name = os.path.basename(file_path)
                self.progress_queue.put((i/total_files*100, f"Processing {file_name}..."))
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    
                    result = {
                        "file_name": file_name,
                        "file_path": file_path,
                        "text_length": len(text),
                        "word_count": len(text.split())
                    }
                    
                    # Perform selected analyses
                    if self.option_vars["extract_summary"].get():
                        result["summary"] = self.processor.generate_summary(text, self.summary_length.get())
                    
                    if self.option_vars["extract_key_phrases"].get():
                        result["key_phrases"] = self.processor.extract_key_phrases(text)
                    
                    if self.option_vars["extract_entities"].get():
                        result["entities"] = self.processor.extract_entities(text)
                    
                    if self.option_vars["extract_topics"].get():
                        result["topics"] = self.processor.extract_topics(text)
                    
                    self.results.append(result)
                    
                except Exception as e:
                    self.results.append({
                        "file_name": file_name,
                        "file_path": file_path,
                        "error": str(e)
                    })
            
            self.progress_queue.put((100, "Processing complete"))
            
            # Update UI in the main thread
            self.root.after(0, self.display_results)
            
        except Exception as e:
            self.progress_queue.put((0, f"Error: {str(e)}"))
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))
    
    def display_results(self):
        """Display the results in the results tab"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            if widget != self.results_frame.winfo_children()[-1]:  # Keep export frame
                widget.destroy()
        
        if not self.results:
            ttk.Label(self.results_frame, text="No results to display").pack()
            self.process_button.config(state=tk.NORMAL)
            return
        
        # Create a results notebook for each file
        results_notebook = ttk.Notebook(self.results_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        for result in self.results:
            file_tab = ttk.Frame(results_notebook, padding=10)
            results_notebook.add(file_tab, text=result["file_name"])
            
            # Check if there was an error
            if "error" in result:
                ttk.Label(file_tab, text=f"Error processing file: {result['error']}", foreground="red").pack(anchor=tk.W)
                continue
            
            # File info section
            info_frame = ttk.LabelFrame(file_tab, text="File Information", padding=5)
            info_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(info_frame, text=f"File: {result['file_name']}").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Path: {result['file_path']}").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Size: {result['text_length']} characters").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Word Count: {result['word_count']} words").pack(anchor=tk.W)
            
            # Create inner notebook for different analysis results
            analysis_notebook = ttk.Notebook(file_tab)
            analysis_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # Summary tab
            if "summary" in result:
                summary_tab = ttk.Frame(analysis_notebook, padding=5)
                analysis_notebook.add(summary_tab, text="Summary")
                
                summary_text = scrolledtext.ScrolledText(summary_tab, wrap=tk.WORD, height=10)
                summary_text.pack(fill=tk.BOTH, expand=True)
                summary_text.insert(tk.END, result["summary"])
                summary_text.config(state=tk.DISABLED)
            
            # Key Phrases tab
            if "key_phrases" in result:
                phrases_tab = ttk.Frame(analysis_notebook, padding=5)
                analysis_notebook.add(phrases_tab, text="Key Phrases")
                
                phrases_text = scrolledtext.ScrolledText(phrases_tab, wrap=tk.WORD, height=10)
                phrases_text.pack(fill=tk.BOTH, expand=True)
                phrases_text.insert(tk.END, "\n".join(result["key_phrases"]))
                phrases_text.config(state=tk.DISABLED)
            
            # Entities tab
            if "entities" in result:
                entities_tab = ttk.Frame(analysis_notebook, padding=5)
                analysis_notebook.add(entities_tab, text="Named Entities")
                
                entities_text = scrolledtext.ScrolledText(entities_tab, wrap=tk.WORD, height=10)
                entities_text.pack(fill=tk.BOTH, expand=True)
                
                for entity_type, entities in result["entities"].items():
                    entities_text.insert(tk.END, f"{entity_type}:\n")
                    entities_text.insert(tk.END, ", ".join(entities) + "\n\n")
                
                entities_text.config(state=tk.DISABLED)
            
            # Topics tab
            if "topics" in result:
                topics_tab = ttk.Frame(analysis_notebook, padding=5)
                analysis_notebook.add(topics_tab, text="Topics")
                
                topics_text = scrolledtext.ScrolledText(topics_tab, wrap=tk.WORD, height=10)
                topics_text.pack(fill=tk.BOTH, expand=True)
                
                for i, topic in enumerate(result["topics"]):
                    topics_text.insert(tk.END, f"Topic {i+1}: {', '.join(topic)}\n\n")
                
                topics_text.config(state=tk.DISABLED)
        
        # Switch to the results tab
        self.notebook.select(1)
        
        # Re-enable the process button
        self.process_button.config(state=tk.NORMAL)
    
    def export_results(self):
        """Export results to CSV"""
        if not self.results:
            tk.messagebox.showwarning("No Results", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Results"
        )
        
        if not file_path:
            return
        
        try:
            # Create a flat structure for the CSV
            flat_results = []
            
            for result in self.results:
                flat_result = {
                    "file_name": result["file_name"],
                    "file_path": result["file_path"],
                    "text_length": result.get("text_length", ""),
                    "word_count": result.get("word_count", ""),
                    "summary": result.get("summary", ""),
                    "key_phrases": ", ".join(result.get("key_phrases", [])),
                }
                
                # Handle entities
                if "entities" in result:
                    for entity_type, entities in result["entities"].items():
                        flat_result[f"entities_{entity_type}"] = ", ".join(entities)
                
                # Handle topics
                if "topics" in result:
                    for i, topic in enumerate(result["topics"]):
                        flat_result[f"topic_{i+1}"] = ", ".join(topic)
                
                flat_results.append(flat_result)
            
            # Convert to DataFrame and export
            df = pd.DataFrame(flat_results)
            df.to_csv(file_path, index=False)
            
            tk.messagebox.showinfo("Export Complete", f"Results exported to {file_path}")
            
        except Exception as e:
            tk.messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentScanner(root)
    root.mainloop()
