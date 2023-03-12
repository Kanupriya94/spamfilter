import requests
import io
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Define the URLs to the spam and ham folders
spam_url = "https://github.com/Kanupriya94/spamfilter/raw/main/spam"
ham_url = "https://github.com/Kanupriya94/spamfilter/raw/main/ham"

# Download the PDF files and create a list of documents
spam_documents = []
for filename in ["spam1.pdf", "spam2.pdf", "spam3.pdf"]:
    response = requests.get(f"{spam_url}/{filename}")
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    content = ""
    for i in range(len(pdf_reader.pages)):
        content += pdf_reader.pages[i].extract_text()
    spam_documents.append(content)

ham_documents = []
for filename in ["ham1.pdf", "ham2.pdf", "ham3.pdf"]:
    response = requests.get(f"{ham_url}/{filename}")
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    content = ""
    for i in range(len(pdf_reader.pages)):
        content += pdf_reader.pages[i].extract_text()
    ham_documents.append(content)

# Vectorize the documents using the bag of words approach
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(spam_documents + ham_documents)

# Create the labels for the spam and ham documents
y = [1]*len(spam_documents) + [0]*len(ham_documents)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model using the Naive Bayes algorithm
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the model's performance
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
