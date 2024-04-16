import numpy as np
# Define the embedding matrix, target word index, and context word indices
# with the provided values for vocabulary size, vector size, and window size.

# Set the parameters
vocab_size = 1000  # Number of words in the vocabulary
vector_size = 30   # Size of the vector for each word
window_size = 4     # The value of N for the context window

# Initialize random vectors for an embedding matrix example
np.random.seed(42)  # Seed for reproducibility
embedding_matrix = np.random.rand(vocab_size, vector_size)  # Embedding matrix for the vocabulary

# Simulate a target word index (for example purposes, we choose an arbitrary index)
target_word_index = np.random.randint(low=0, high=vocab_size)

# Simulate context word indices (for example purposes, we randomly select indices)
# In a real scenario, these would come from the actual words in the context of the target word in a sentence
context_word_indices = np.random.randint(low=0, high=vocab_size, size=window_size * 2)

# Correct implementation of the skip-gram loss calculation as per the formula provided
def skip_gram_loss_corrected(embedding_matrix, target_word_index, context_word_indices):
    # Get the vector for the target word
    v_wi = embedding_matrix[target_word_index]
    
    # Compute the softmax denominator once for efficiency
    softmax_denominator = np.sum(np.exp(embedding_matrix @ v_wi))
    
    # Initialize loss
    loss = 0
    
    # Loop through each context word index
    for context_word_index in context_word_indices:
        # Get the vector for the context word
        v_wj = embedding_matrix[context_word_index]
        
        # Calculate the dot product between the target and context word vectors
        dot_product = np.dot(v_wi, v_wj)
        
        # Update the loss by subtracting the log of softmax denominator from the dot product
        loss += (dot_product - np.log(softmax_denominator))
    
    return loss

# Calculate the loss for our simulated target word and context words
loss_corrected = skip_gram_loss_corrected(embedding_matrix, target_word_index, context_word_indices)
loss_corrected, target_word_index, context_word_indices
