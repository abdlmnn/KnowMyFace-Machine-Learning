import sqlite3
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Connect to SQLite
conn = sqlite3.connect('dataset.db')
cursor = conn.cursor()

# Query all the data from the faces table
cursor.execute("SELECT name, embedding FROM faces")
data = cursor.fetchall()

# Prepare true and predicted identities (example data, adjust as per your project)
true_labels = []  # Replace with your actual true labels
predicted_labels = []  # Replace with your actual predicted labels

# Assuming we have some method to get predicted labels (e.g., based on a classifier or model)
for name, embedding in data:
    # Here, you should implement your prediction logic based on the embeddings
    # For simplicity, we'll assume the predicted label matches the true label
    true_labels.append(name)  # Replace with actual true labels
    predicted_labels.append(name)  # Replace with actual predicted labels

# Evaluate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')  # Weighted average for multi-class
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Close the database connection
conn.close()
