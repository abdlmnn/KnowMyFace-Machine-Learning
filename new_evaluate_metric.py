import sqlite3
import pickle
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
from collections import Counter

# Connect to your database
conn = sqlite3.connect('test.db')  # Use your actual database file
cursor = conn.cursor()

# Get all stored faces
cursor.execute("SELECT name, embedding FROM faces")
data = cursor.fetchall()
conn.close()

# For evaluation, you need true labels and predicted labels
# Since your system uses similarity matching, we'll simulate evaluation
# In a real scenario, you'd have test images with known labels

# Example: Create mock test data (replace with actual test data)
# Assuming you have test embeddings and their true labels
test_embeddings = []  # List of test face embeddings
true_labels = []      # True names for test images

# For demonstration, let's use your stored data as "test data"
# In practice, you'd separate training and test sets
for name, emb_blob in data:
    embedding = pickle.loads(emb_blob)
    test_embeddings.append(embedding)
    true_labels.append(name)

# Function to predict labels using your recognition logic
def predict_labels(test_embeddings, stored_data, threshold=0.32):
    predictions = []
    
    # Recreate stored faces dict (same as in your app)
    stored_faces = {}
    for stored_name, stored_emb in stored_data:
        emb = pickle.loads(stored_emb)
        stored_faces.setdefault(stored_name, []).append(emb)
    
    for test_emb in test_embeddings:
        predicted_name = "Unknown"
        min_dist = 1.0
        
        # Compare with all stored embeddings
        for name, embs in stored_faces.items():
            # Use cosine similarity (1 - cosine distance)
            similarities = [np.dot(test_emb, e) / (np.linalg.norm(test_emb) * np.linalg.norm(e)) 
                          for e in embs if e is not None]
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity > (1 - threshold) and avg_similarity > min_dist:
                    min_dist = avg_similarity
                    predicted_name = name
        
        predictions.append(predicted_name)
    
    return predictions

# Get predictions
predicted_labels = predict_labels(test_embeddings, data)

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)

# For multi-class classification with imbalanced classes
precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

precision_weighted = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
recall_weighted = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1_weighted = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

# Get per-class metrics
precision_per_class = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
recall_per_class = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)

# Calculate support (number of true instances for each class)
support = Counter(true_labels)
support_list = [support[label] for label in sorted(support.keys())]

# Print results
print("=== FACE RECOGNITION EVALUATION METRICS ===")
print(f"Accuracy: {accuracy:.4f}")
print()

print("Macro Average:")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall: {recall_macro:.4f}")
print(f"  F1-Score: {f1_macro:.4f}")
print()

print("Weighted Average:")
print(f"  Precision: {precision_weighted:.4f}")
print(f"  Recall: {recall_weighted:.4f}")
print(f"  F1-Score: {f1_weighted:.4f}")
print()

print("Per-Class Metrics:")
print("Class\t\tPrecision\tRecall\t\tF1-Score\tSupport")
print("-" * 70)
for i, class_name in enumerate(sorted(support.keys())):
    print(f"{class_name}\t\t{precision_per_class[i]:.4f}\t\t{recall_per_class[i]:.4f}\t\t{f1_per_class[i]:.4f}\t\t{support_list[i]}")

print()
print("Detailed Classification Report:")
print(classification_report(true_labels, predicted_labels, zero_division=0))

print()
print("Confusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)
