import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse
from PepDiffusion.src.TransVAE import *  # Import the TransVAE model
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def load_peptides(file_path):
    """Load peptides from a file where each line contains one peptide sequence."""
    with open(file_path, 'r') as f:
        peptides = [line.strip() for line in f if line.strip()]
    # Convert to numpy array with shape (n_peptides, 1)
    return np.array(peptides).reshape(-1, 1)

def embed_peptides(peptides, model, batch_size=512, device='cuda'):
    """Embed peptides using the VAE model."""
    # Convert peptides to model input format
    data = vae_data_gen(peptides, params["src_len"], char_dict=w2i)
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding peptides"):
            mols_data = batch[:,:-1].to(device)
            src = Variable(mols_data).long()
            src_mask = (src != w2i["_"]).unsqueeze(-2)
            # Get the latent representation
            _, mem, _, _ = model.encode(src, src_mask)
            embeddings.append(mem.cpu().numpy())
    
    return np.vstack(embeddings)

def cluster_embeddings(embeddings, n_clusters=5):
    """Perform clustering on the embeddings."""
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    return clusters, reduced_embeddings

def visualize_clusters(reduced_embeddings, clusters, peptides, output_file='clusters.png', method='Baseline'):
    """Visualize the clusters in 2D space."""
    plt.figure(figsize=(12, 8))
    
    # Create a colormap with distinct colors for each cluster
    n_clusters = len(np.unique(clusters))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))  # Using tab10 colormap for distinct colors
    
    # Plot each cluster separately with its own color
    for i in range(n_clusters):
        mask = clusters == i
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6)
    
    # Add some peptide labels
    for i, peptide in enumerate(peptides):
        if i % 10 == 0:  # Label every 10th peptide to avoid overcrowding
            plt.annotate(peptide, 
                        (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                        fontsize=8)
    
    plt.legend(title='Clusters')
    plt.title(f'{method}-Generated Peptide Clusters (PCA Projection)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(output_file)
    plt.close()

def find_optimal_clusters(embeddings, max_clusters=10):
    """Find the optimal number of clusters using the Elbow Method."""
    distortions = []
    K = range(1, max_clusters + 1)
    
    for k in tqdm(K, desc="Finding optimal clusters"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('elbow_curve.png')
    plt.close()
    
    # Find the elbow point using the second derivative
    distortions = np.array(distortions)
    second_derivative = np.gradient(np.gradient(distortions))
    optimal_k = np.argmax(second_derivative) + 1
    
    # Ensure we have at least 2 clusters
    optimal_k = max(2, optimal_k)
    
    # Calculate silhouette scores for different k values
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for different k values')
    plt.savefig('silhouette_scores.png')
    plt.close()
    
    # Find k with highest silhouette score
    best_k = np.argmax(silhouette_scores) + 2  # +2 because we start from k=2
    
    print(f"\nCluster Analysis:")
    print(f"Elbow method suggests k = {optimal_k}")
    print(f"Silhouette analysis suggests k = {best_k}")
    
    # Use the maximum of the two methods to ensure we have enough clusters
    final_k = max(optimal_k, best_k)
    print(f"Using k = {final_k}")
    
    return final_k

def calculate_cluster_metrics(embeddings, clusters):
    """Calculate various metrics to evaluate cluster quality."""
    metrics = {}
    
    # Only calculate metrics if we have more than 1 cluster
    if len(np.unique(clusters)) > 1:
        # Silhouette Score (-1 to 1, higher is better)
        metrics['silhouette_score'] = silhouette_score(embeddings, clusters)
        
        # Calinski-Harabasz Score (higher is better)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, clusters)
        
        # Davies-Bouldin Score (lower is better)
        metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, clusters)
        
        # Calculate average distance between cluster centers
        kmeans = KMeans(n_clusters=len(np.unique(clusters)), random_state=42)
        kmeans.fit(embeddings)
        centers = kmeans.cluster_centers_
        
        # Calculate pairwise distances between cluster centers
        center_distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance = np.linalg.norm(centers[i] - centers[j])
                center_distances.append(distance)
        
        metrics['avg_cluster_center_distance'] = np.mean(center_distances)
        metrics['min_cluster_center_distance'] = np.min(center_distances)
    else:
        metrics['silhouette_score'] = None
        metrics['calinski_harabasz_score'] = None
        metrics['davies_bouldin_score'] = None
        metrics['avg_cluster_center_distance'] = None
        metrics['min_cluster_center_distance'] = None
    
    return metrics

def analyze_single_cluster_hypothesis(embeddings, clusters):
    """Analyze if the clustering is effectively just one cluster."""
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    print("\nSingle Cluster Analysis:")
    print(f"Number of unique clusters found: {n_clusters}")
    
    # Count points in each cluster
    cluster_counts = np.bincount(clusters)
    print("\nPoints per cluster:")
    for i in range(n_clusters):
        print(f"Cluster {i}: {cluster_counts[i]} points")
    
    # Calculate average distance to cluster center
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    centers = kmeans.cluster_centers_
    
    avg_distances = []
    for i in range(n_clusters):
        cluster_points = embeddings[clusters == i]
        distances = np.linalg.norm(cluster_points - centers[i], axis=1)
        avg_distances.append(np.mean(distances))
    
    print("\nAverage distance to cluster center:")
    for i in range(n_clusters):
        print(f"Cluster {i}: {avg_distances[i]:.3f}")
    
    # If we have more than one cluster, calculate between-cluster distances
    if n_clusters > 1:
        between_cluster_distances = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                distance = np.linalg.norm(centers[i] - centers[j])
                between_cluster_distances.append(distance)
        
        print(f"\nAverage distance between cluster centers: {np.mean(between_cluster_distances):.3f}")
        print(f"Minimum distance between cluster centers: {np.min(between_cluster_distances):.3f}")
    
    return n_clusters == 1

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cluster peptide sequences using VAE embeddings')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained VAE model')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input file containing peptide sequences')
    parser.add_argument('--output_file', type=str, default='clustering_results.csv',
                      help='Path to save clustering results (default: clustering_results.csv)')
    parser.add_argument('--n_clusters', type=int, default=None,
                      help='Number of clusters (if not specified, will be determined automatically)')
    parser.add_argument('--max_clusters', type=int, default=10,
                      help='Maximum number of clusters to consider when finding optimal k (default: 10)')
    parser.add_argument('--plot_file', type=str, default='clusters.png',
                      help='Path to save cluster visualization (default: clusters.png)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run the model on (cuda/cpu)')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load the VAE model
    model = create_VAE()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load peptides
    peptides = load_peptides(args.input_file)
    
    # Get embeddings
    print("Embedding peptides...")
    embeddings = embed_peptides(peptides, model, device=device)
    
    # Determine optimal number of clusters if not specified
    if args.n_clusters is None:
        print("Finding optimal number of clusters...")
        optimal_k = find_optimal_clusters(embeddings, max_clusters=args.max_clusters)
        print(f"Optimal number of clusters: {optimal_k}")
        n_clusters = optimal_k
    else:
        n_clusters = args.n_clusters
    
    # Perform clustering
    print("Clustering embeddings...")
    clusters, reduced_embeddings = cluster_embeddings(embeddings, n_clusters=n_clusters)
    
    # Analyze if the clustering is effectively just one cluster
    is_single_cluster = analyze_single_cluster_hypothesis(embeddings, clusters)
    
    # Calculate and print cluster metrics
    print("\nCalculating cluster metrics...")
    metrics = calculate_cluster_metrics(embeddings, clusters)
    print("\nCluster Quality Metrics:")
    if metrics['silhouette_score'] is not None:
        print(f"Silhouette Score: {metrics['silhouette_score']:.3f} (higher is better, range: -1 to 1)")
        print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f} (higher is better)")
        print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f} (lower is better)")
        print(f"Average Distance Between Cluster Centers: {metrics['avg_cluster_center_distance']:.3f}")
        print(f"Minimum Distance Between Cluster Centers: {metrics['min_cluster_center_distance']:.3f}")
    else:
        print("Warning: Could not calculate cluster metrics as only one cluster was found.")
        print("Consider using a different number of clusters or checking your data for natural groupings.")
    
    # Visualize results
    print("\nVisualizing clusters...")
    method = 'Baseline (Genetic Algorithm)'
    visualize_clusters(reduced_embeddings, clusters, peptides.flatten(), 
                      output_file=args.plot_file, method=method)
    
    # Save results
    results_df = pd.DataFrame({
        'peptide': peptides.flatten(),  # Flatten the peptides array
        'cluster': clusters
    })
    results_df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 


# baseline
# Using k = 10
# Cluster Quality Metrics:
# Silhouette Score: 0.347 (higher is better, range: -1 to 1)
# Calinski-Harabasz Score: 11.822 (higher is better)
# Davies-Bouldin Score: 0.712 (lower is better)
# Average Distance Between Cluster Centers: 12.205
# Minimum distance Between Cluster Centers: 6.809

# VAE
# Using k = 2
# DecoderCluster Quality Metrics:
# Silhouette Score: 0.499 (higher is better, range: -1 to 1)
# Calinski-Harabasz Score: 360.824 (higher is better)
# Davies-Bouldin Score: 1.089 (lower is better)
# Average Distance Between Cluster Centers: 9.395
# Minimum Distance Between Cluster Centers: 9.395