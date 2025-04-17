from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained Sentence-BERT model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Sample job titles from different domains
job_titles = [
    "Software Engineer",
    "Data Scientist",
    "Machine Learning Engineer",
    "Doctor",
    "Nurse",
    "Clinical Researcher",
    "Teacher",
    "Professor",
    "Accountant",
    "Financial Analyst",
    "Graphic Designer",
    "Marketing Specialist",
    "Sales Manager",
    "Customer Support Representative",
    "Electrician",
    "Mechanical Engineer"
]

# Labels for job domains (for visualization)
domains = [
    "Tech", "Tech", "Tech",
    "Healthcare", "Healthcare", "Healthcare",
    "Education", "Education",
    "Finance", "Finance",
    "Creative", "Creative",
    "Business", "Business",
    "Trades", "Trades"
]

# Generate embeddings for job titles
embeddings = model.encode(job_titles)

# Apply UMAP to reduce embeddings to 2D
umap_embeddings = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine').fit_transform(embeddings)

# Create a scatter plot
plt.figure(figsize=(10, 7))
for i, domain in enumerate(set(domains)):
    idxs = np.where(np.array(domains) == domain)[0]
    plt.scatter(umap_embeddings[idxs, 0], umap_embeddings[idxs, 1], label=domain)

# Add job titles to the plot
for i, title in enumerate(job_titles):
    plt.text(umap_embeddings[i, 0] + 0.02, umap_embeddings[i, 1], title, fontsize=9)

plt.title("Job Titles Clustered by Embeddings")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.show()

