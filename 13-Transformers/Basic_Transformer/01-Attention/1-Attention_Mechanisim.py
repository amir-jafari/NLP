import numpy as np
from scipy.special import softmax
# ----------------------------------------------------------------------------------------------------------------------
print("Step 1: Input : 3 inputs, d_model=4")
x =np.array([[1.0, 0.0, 1.0, 0.0],   # Input 1
             [0.0, 2.0, 0.0, 2.0],   # Input 2
             [1.0, 1.0, 1.0, 1.0]])  # Input 3
print(x)
# ----------------------------------------------------------------------------------------------------------------------
print("Step 2: weights 3 dimensions x d_model=4")
print("w_query")
w_query =np.array([[1, 0, 1],
                   [1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 1]])
print(w_query)
# ----------------------------------------------------------------------------------------------------------------------
print("w_key")
w_key =np.array([[0, 0, 1],
                 [1, 1, 0],
                 [0, 1, 0],
                 [1, 1, 0]])
print(w_key)

# ----------------------------------------------------------------------------------------------------------------------
print("w_value")
w_value = np.array([[0, 2, 0],
                    [0, 3, 0],
                    [1, 0, 3],
                    [1, 1, 0]])
print(w_value)
# ----------------------------------------------------------------------------------------------------------------------
print("Step 3: Matrix multiplication to obtain Q,K,V")

print("Queries: x * w_query")
Q=np.matmul(x,w_query)
print(Q)
# ----------------------------------------------------------------------------------------------------------------------
print("Step 3: Matrix multiplication to obtain Q,K,V")

print("Keys: x * w_key")
K=np.matmul(x,w_key)
print(K)
# ----------------------------------------------------------------------------------------------------------------------
print("Values: x * w_value")
V=np.matmul(x,w_value)
print(V)
# ----------------------------------------------------------------------------------------------------------------------
print("Step 4: Scaled Seq2Seq Scores")
k_d=1
attention_scores = (Q @ K.transpose())/k_d
print(attention_scores)
# ----------------------------------------------------------------------------------------------------------------------
print("Step 5: Scaled softmax attention_scores for each vector")
attention_scores[0]=softmax(attention_scores[0])
attention_scores[1]=softmax(attention_scores[1])
attention_scores[2]=softmax(attention_scores[2])
print(attention_scores[0])
print(attention_scores[1])
print(attention_scores[2])
# ----------------------------------------------------------------------------------------------------------------------
print("Step 6: attention value obtained by score1/k_d * V")
print(V[0])
print(V[1])
print(V[2])
print("Seq2Seq 1")
attention1=attention_scores[0].reshape(-1,1)
attention1=attention_scores[0][0]*V[0]
print(attention1)

print("Seq2Seq 2")
attention2=attention_scores[0][1]*V[1]
print(attention2)

print("Seq2Seq 3")
attention3=attention_scores[0][2]*V[2]
print(attention3)
# ----------------------------------------------------------------------------------------------------------------------
print("Step 7: summed the results to create the first line of the output matrix")
attention_input1=attention1+attention2+attention3
print(attention_input1)
# ----------------------------------------------------------------------------------------------------------------------
print("Step 8: Step 1 to 7 for inputs 1 to 3")
'''
We assume we have 3 results with learned weights (they were not trained in this example)
We assume we are implementing the original Transformer paper. We will have 3 results of 64 dimensions each
'''
attention_head1=np.random.random((3, 64))
print(attention_head1)
# ----------------------------------------------------------------------------------------------------------------------
print("Step 9: We assume we have trained the 8 heads of the attention sub-layer")
z0h1=np.random.random((3, 64))
z1h2=np.random.random((3, 64))
z2h3=np.random.random((3, 64))
z3h4=np.random.random((3, 64))
z4h5=np.random.random((3, 64))
z5h6=np.random.random((3, 64))
z6h7=np.random.random((3, 64))
z7h8=np.random.random((3, 64))
print("shape of one head",z0h1.shape,"dimension of 8 heads",64*8)
# ----------------------------------------------------------------------------------------------------------------------
print("Step 10: Concatenation of heads 1 to 8 to obtain the original 8x64=512 output dimension of the model")
output_attention=np.hstack((z0h1,z1h2,z2h3,z3h4,z4h5,z5h6,z6h7,z7h8))
print(output_attention)
# ----------------------------------------------------------------------------------------------------------------------
from transformers import pipeline
translator = pipeline("translation_en_to_fr")
print(translator("It is easy to translate languages with transformers", max_length=40))

