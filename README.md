# Transformer
Transformer Neural Networks architecture implementation

- **Vanilla Transformer** - based on [Attention is all you need](https://arxiv.org/abs/1706.03762) paper with Encoder-Decoder architecture
- **Graph Transformer** - extend idea of Transformer to Graph Neural Networks. Encoder only architecture. 
Attention is only self-attention.  
Attention mechanism is extended to support attention from nodes to edge features and vice versa.
There is no masking in this implementation. The mask is applied explicitly when nodes are multiplied by edge features.   
Model is returning processed results for both nodes and edge features.
