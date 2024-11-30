import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def attention(query, key, value):
    scores = np.dot(query, key.T) / np.sqrt(key.shape[-1])
    weights = softmax(scores)
    return np.dot(weights, value)

def multi_head_attention(queries, keys, values, num_heads, embed_dim):
    head_dim = embed_dim // num_heads
    outputs = []
    for _ in range(num_heads):
        q = queries @ np.random.randn(embed_dim, head_dim)
        k = keys @ np.random.randn(embed_dim, head_dim)
        v = values @ np.random.randn(embed_dim, head_dim)
        outputs.append(attention(q, k, v))
    return np.concatenate(outputs, axis=-1)

def feedforward(x, embed_dim, ff_dim):
    w1 = np.random.randn(embed_dim, ff_dim)
    w2 = np.random.randn(ff_dim, embed_dim)
    return np.maximum(0, x @ w1) @ w2

def add_and_norm(x, sublayer_output, epsilon=1e-6):
    return (x + sublayer_output - np.mean(x + sublayer_output, axis=-1, keepdims=True)) / \
           (np.std(x + sublayer_output, axis=-1, keepdims=True) + epsilon)

def positional_encoding(seq_length, embed_dim):
    positions = np.arange(seq_length)[:, np.newaxis]
    dims = np.arange(embed_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / embed_dim)
    angles = positions * angle_rates
    encoding = np.zeros((seq_length, embed_dim))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding

class Transformer:
    def __init__(self, vocab_size, seq_length, embed_dim, num_heads, num_layers, ff_dim):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.embedding = np.random.randn(vocab_size, embed_dim)
        self.positional_encoding = positional_encoding(seq_length, embed_dim)

    def forward(self, x):
        x = self.embedding[x] + self.positional_encoding
        for _ in range(self.num_layers):
            attention_output = multi_head_attention(x, x, x, self.num_heads, self.embed_dim)
            x = add_and_norm(x, attention_output)
            ff_output = feedforward(x, self.embed_dim, self.ff_dim)
            x = add_and_norm(x, ff_output)
        return softmax(x @ self.embedding.T)

    def generate(self, prompt, max_length):
        sequence = prompt
        for _ in range(max_length - len(prompt)):
            output = self.forward(sequence)
            next_token = np.argmax(output[-1])
            sequence = np.append(sequence, next_token)
        return sequence

# Example Usage
if __name__ == "__main__":
    vocab_size = 50
    seq_length = 10
    embed_dim = 16
    num_heads = 2
    num_layers = 2
    ff_dim = 32

    transformer = Transformer(vocab_size, seq_length, embed_dim, num_heads, num_layers, ff_dim)
    prompt = np.array([1, 2, 3, 4])
    generated_sequence = transformer.generate(prompt, max_length=10)
    print("Generated Sequence:", generated_sequence)
