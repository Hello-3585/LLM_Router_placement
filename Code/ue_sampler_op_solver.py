import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix


HF_token ="YOUR HF token"

class Initial_guess:
    def __init__(self):
        self.llama_model =None
        self.tokenizer = None
    def config(self):
        use_8bit = True
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        # Load the Llama 3 8B model and tokenizer
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llama_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                   quantization_config=bnb_config,
                                                   device_map="auto",
                                                   token=HF_token)
    def generate(self,request):
        system={"role": "system", 
            "content": 
        """You are an advanced telecommunications network clustering assistant. Your task is to generate an initial guess for the cluster coordinates in a telecommunications network optimization problem based on the user's request. 
        The number of clusters is always equal to the number of routers provided in the input. Your response must adhere to the following requirements:
        Understand the Request: Interpret the user's description of the network environment and constraints, including any explicit or implicit optimization objectives (e.g., reducing distance, balancing load, maximizing coverage, or minimizing interference).
        Generate Coordinates: Based on the problem description, produce a structured output containing the initial guess for the coordinates of the clusters. The output should be in a clear, machine-readable format.
        Adhere to Constraints: If specific constraints (such as a physical area, minimum separation, or maximum coverage) are provided, ensure the coordinates satisfy these conditions.

        Here is a Mapping of the Optimisation Functions with their Objectives:
        1. KM (K-Means): Distance Reduction
        2. WKHM (Weighted K-Harmonic Means): Balancing Load and Power Loss Optimization
        3. CKM (Constrained K-Means): Load Balancing 
        4. KC (K-Centers): Maximize Minimum RSRP[Received Signal Reference Power]
        5. KHM (K-Harmonic Means): Power Loss Optimization/ Maximising Received Signal Reference Power

        Use techniques like K-Means to ensure logical spacing of clusters and ensure that all cluster centers lie within the area constraints.
        Your Output Should be of the following JSON format:
        {cluster 1: [x1,y1],
        cluster 2: [x2,y2],
        .
        .
        .
        cluster N: [xN,yn]}
        Where N is the number of Routers

        START YOUR RESPONSE WITH A ^
        DO NOT PROVIDE WITH ANY REASONING JUST RETURN THE JSONISED OUTPUT.
        ONLY PROVIDE X and Y coordinates For Each Cluster
        RETURNING ANYTHING ELSE APART FROM THE JSON OUTPUT WILL BE PENALISED
        """
        }
        self.config()
        message=[system,
            {"role":"user","content":request}]
        input_ids = self.tokenizer.apply_chat_template(message, return_tensors="pt",padding=True, truncation=True,).to(self.llama_model.device)
            
        with torch.no_grad():
            output = self.llama_model.generate(input_ids, max_length=input_ids.shape[1]+150, num_return_sequences=1, temperature=1);
            
        modified_prompt = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = modified_prompt.split("^")[-1].strip()
        return answer

class Generate_Samples:
    def uniform_sampling(self,low, high, num_samples):
        samples = np.random.uniform(low, high, num_samples)
        return samples
    def gaussian_sampling_2d(self,x_mean, x_std, num_samples):
        x_samples = np.random.normal(x_mean, x_std, num_samples)
        return x_samples
    def exponential_sampling_2d(self,x_low, x_high ,scale_x, num_samples):
        x_samples = np.random.exponential(scale_x, num_samples)
        x_samples = x_low + (x_samples / x_samples.max()) * (x_high - x_low)
        return x_samples
    def bi_exponential_sampling_2d(self,x_low, x_high, scale_x, num_samples):
        num_samples_half = num_samples // 2
        x_samples_1 = np.random.laplace(loc=x_low, scale=scale_x, size=num_samples_half)
        x_samples_2 = np.random.laplace(loc=x_high, scale=scale_x, size=num_samples_half)
        x_samples = np.concatenate([x_samples_1, x_samples_2])
        x_samples = np.clip(x_samples, x_low, x_high)
        
        return x_samples

class OP_SOLVERS:
    def k_means_clustering(self,X, K):
        kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
        return kmeans.cluster_centers_, kmeans.labels_

    def constrained_k_means(self,data, K, Nmin, Nmax, max_iter=300, tol=1e-4):
        # Initialize centroids randomly from the data points
        centroids = data[np.random.choice(range(len(data)), K, replace=False)]
        # Initialize labels and previous labels
        labels = np.zeros(len(data))
        prev_labels = np.zeros(len(data))
        distances = np.zeros((data.shape[0], K))
        
        for iteration in range(max_iter):
            # Calculate distances from data points to centroids
            distances = cdist(data, centroids, 'euclidean')
            
            # Assign data points to nearest centroid
            temp_labels = np.argmin(distances, axis=1)
            
            # Enforce constraints
            for i in range(K):
                cluster_indices = np.where(temp_labels == i)[0]
                cluster_size = len(cluster_indices)
                if cluster_size < Nmin:
                    # Too few elements, randomly sample additional points to add
                    additional_points = np.random.choice(np.delete(np.arange(len(data)), cluster_indices), 
                                                        Nmin - cluster_size, replace=False)
                    temp_labels[additional_points] = i
                elif cluster_size > Nmax:
                    # Too many elements, randomly select points to remove
                    points_to_remove = np.random.choice(cluster_indices, cluster_size - Nmax, replace=False)
                    # Set their labels to an invalid number so they can be reassigned
                    temp_labels[points_to_remove] = -1
            
            # Reassign removed points to the nearest valid cluster
            removed_indices = np.where(temp_labels == -1)[0]
            if len(removed_indices) > 0:
                dist_to_valid_centroids = cdist(data[removed_indices], centroids, 'euclidean')
                new_labels = np.argmin(dist_to_valid_centroids, axis=1)
                temp_labels[removed_indices] = new_labels

            # Check for convergence (if labels haven't changed)
            if np.all(prev_labels == temp_labels):
                break
            prev_labels = temp_labels.copy()
            
            # Calculate new centroids
            for i in range(K):
                if np.any(temp_labels == i):
                    centroids[i] = np.mean(data[temp_labels == i], axis=0)

            # Check for small movements
            if iteration > 0 and max(np.linalg.norm(centroids - old_centroids, axis=1)) < tol:
                break
            old_centroids = centroids.copy()

        return centroids, temp_labels
    def KHM(self,X, K, p=2, max_iter=10000, epsilon=1e-8,initial = None):
        # Step 1: Initialize K random centroids
        n, d = X.shape
        indices = np.random.choice(n, K, replace=False)
        if initial is not None:
            centroids = initial
        else:
            centroids = X[indices].copy()
        
        def calculate_distances(X, centroids):
            distances = np.zeros((X.shape[0], len(centroids)))
            for j, centroid in enumerate(centroids):
                # Calculate squared Euclidean distance
                diff = X - centroid
                distances[:, j] = np.sum(diff**2, axis=1)
            return np.sqrt(distances) + epsilon  # Take square root and avoid division by zero
        
        prev_objective = float('inf')
        
        for iteration in range(max_iter):
            # Step 2: Calculate distances and objective function
            distances = calculate_distances(X, centroids)
            
            # Calculate harmonics according to equation (1)
            inv_distances = 1/distances**p
            harmonic_means = K / np.sum(inv_distances, axis=1)
            current_objective = np.sum(harmonic_means)
            if iteration == 0: print(current_objective/n)
            
            # Check convergence
            if abs(prev_objective - current_objective) < epsilon:
                break
            prev_objective = current_objective
            
            # Step 3: Calculate memberships and weights
            # Membership function m(C_j|x_i) - Equation (2)
            inv_dist_p = 1/distances**p
            memberships = inv_dist_p / np.sum(inv_dist_p, axis=1)[:, np.newaxis]
            
            # Weight function w(x_i) - Equation (3)
            weights = np.sum(inv_dist_p, axis=1) / (np.sum(1/distances, axis=1)**2)
            
            # Step 4: Update centroids - Equation (4)
            new_centroids = np.zeros_like(centroids)
            for j in range(K):
                numerator = np.sum((memberships[:, j] * weights)[:, np.newaxis] * X, axis=0)
                denominator = np.sum(memberships[:, j] * weights)
                new_centroids[j] = numerator / denominator
            
            # Update centroids if they're valid (not NaN)
            if not np.any(np.isnan(new_centroids)):
                centroids = new_centroids
        
        # Final cluster assignments
        final_distances = calculate_distances(X, centroids)
        cluster_assignments = np.argmin(final_distances, axis=1)
        
        return centroids, cluster_assignments, current_objective
    
    def WKHM(self,X, K, p=2, q=2, max_iter=100, epsilon=1e-8):
        """
    """
    # Step 1: Initialize K random centroids
        n, d = X.shape
        indices = np.random.choice(n, K, replace=False)
        centroids = X[indices].copy()
        
        def calculate_distances(X, centroids):
            distances = np.zeros((X.shape[0], len(centroids)))
            for j, centroid in enumerate(centroids):
                distances[:, j] = np.sqrt(np.sum((X - centroid)**2, axis=1))
            return distances + epsilon
        
        prev_objective = float('inf')
        
        for iteration in range(max_iter):
            # Calculate distances
            distances = calculate_distances(X, centroids)
            
            # Calculate soft weights (w_il) according to paper formulation
            # w_il = ||x_i - m_l||^(-q) / sum(||x_i - m_j||^(-q))
            inv_distances_q = distances**(-q)
            soft_weights = inv_distances_q / np.sum(inv_distances_q, axis=1)[:, np.newaxis]
            
            # Calculate weighted denominator: sum(w_il / ||x_i - m_l||^p)
            weighted_denominators = np.sum(soft_weights / distances**p, axis=1)
            
            # Calculate WKHM objective: sum(K / weighted_denominators)
            current_objective = np.sum(K / weighted_denominators)
            
            # Check convergence
            if abs(prev_objective - current_objective) < epsilon:
                break
            prev_objective = current_objective
            
            # Calculate memberships and weights for centroid updates
            inv_distances_p = 1.0 / distances**p
            memberships = inv_distances_p / np.sum(inv_distances_p, axis=1)[:, np.newaxis]
            
            # Calculate weights similar to KHM but incorporating soft weights
            weights = (inv_distances_p * soft_weights) / np.sum((inv_distances_p * soft_weights)**2, axis=1)[:, np.newaxis]
            
            # Update centroids
            for j in range(K):
                combined_weights = weights[:, j] * memberships[:, j]
                numerator = np.sum(combined_weights[:, np.newaxis] * X, axis=0)
                denominator = np.sum(combined_weights)
                centroids[j] = numerator / denominator
        
        # Final cluster assignments
        final_distances = calculate_distances(X, centroids)
        inv_distances_p = 1.0 / final_distances**p
        final_memberships = inv_distances_p / np.sum(inv_distances_p, axis=1)[:, np.newaxis]
        cluster_assignments = np.argmax(final_memberships, axis=1)
        
        return centroids, cluster_assignments, current_objective

    def k_centers(self,X, K):
        n = X.shape[0]
        np.random.seed(0)
        # Randomly choose an initial center
        centers = [np.random.randint(0, n)]
        
        # Greedily choose centers to minimize the maximum distance to any point
        for _ in range(1, K):
            dist_matrix = distance_matrix(X, X[centers])
            min_dist = np.min(dist_matrix, axis=1)  # Find the minimum distance for each point to any center
            next_center = np.argmax(min_dist)  # Choose the next center to be the point with the maximum min distance
            centers.append(next_center)
        
        # Get the final centers positions
        final_centers = X[centers]

        # Compute labels for each data point based on the nearest center
        final_dist_matrix = distance_matrix(X, final_centers)
        labels = np.argmin(final_dist_matrix, axis=1)

        return final_centers, labels

