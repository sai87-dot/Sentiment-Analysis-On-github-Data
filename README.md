Sentiment Analysis of Flipkart Product Reviews
Overview
This project focuses on sentiment analysis of customer reviews for the "YONEX MAVIS 350 Nylon Shuttle" product from Flipkart. By classifying reviews as positive or negative, the project aims to provide actionable insights into customer satisfaction and dissatisfaction. Additionally, it identifies pain points for customers who write negative reviews.

Features
Sentiment Classification: Analyze reviews to determine whether they are positive or negative.
Customer Insights: Understand features contributing to customer satisfaction or dissatisfaction.
Interactive Application: User-friendly web application for real-time sentiment analysis of reviews.
Deployment: Hosted application accessible via the web.
Dataset
Source: Real-time data scraped by Data Engineers from Flipkart.
Size: 8,518 reviews for the "YONEX MAVIS 350 Nylon Shuttle" product.
Features:
Reviewer Name
Rating
Review Title
Review Text
Place of Review
Date of Review
Up Votes
Down Votes
Workflow
Data Loading and Analysis

Explore and understand patterns in the dataset.
Derive insights into customer satisfaction.
Data Preprocessing

Text Cleaning: Remove special characters, punctuation, and stopwords.
Text Normalization: Use lemmatization or stemming to normalize text.
Text Embedding

Experimented with feature extraction methods:
Bag-of-Words (BoW)
Term Frequency-Inverse Document Frequency (TF-IDF)
Word2Vec (W2V)
BERT
Model Training

Trained machine learning and deep learning models on embedded text data.
Optimized hyperparameters for better accuracy.
Model Evaluation

Used F1-Score as the evaluation metric to assess model performance.
Application Development

Developed a web app using Flask or Streamlit for real-time sentiment analysis.
Integrated the trained model into the application.
Deployment

Deployed the application on an AWS EC2 instance for public access.
Testing and Monitoring

Tested the application for errors.
Monitored performance for continuous improvement.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/sentiment-analysis-flipkart-reviews.git
cd sentiment-analysis-flipkart-reviews
Install required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the application locally:

bash
Copy code
streamlit run app.py
OR

bash
Copy code
flask run
Usage
Enter a product review in the application interface.
The application will classify the review as positive or negative.
View insights into product satisfaction or dissatisfaction based on sentiment classification.
Technologies Used
Programming Language: Python
Libraries:
Text Preprocessing: NLTK, SpaCy
Feature Extraction: Scikit-learn, Gensim, Transformers
Model Training: Scikit-learn, TensorFlow, PyTorch
Visualization: Matplotlib, Seaborn
Web Frameworks: Flask, Streamlit
Deployment: AWS EC2
Results
Achieved an F1-Score of X.XX (replace with actual score).
Successfully deployed a real-time sentiment analysis app accessible on the web.
Future Work
Incorporate multilingual support for reviews in different languages.
Implement advanced deep learning models such as GPT or fine-tuned BERT.
Explore additional datasets for broader applicability.
Contributors
