"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        ##我的代码开始
        ## Split the feature dimension into heads without changing token order.
        B, N, D = x.shape
        x_reshaped = x.reshape(B,N,self.heads,self.head_dim)
        ## Move the head axis ahead of the token axis.
        head_tokens = x_reshaped.transpose(0,2,1,3)
        return head_tokens
        ##结束

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        ##我的代码开始
        ## Transpose the last two axes of K for query-key comparisons.
        k_transposed = k.transpose(0,1,3,2)
        ## Use a float32 scale to avoid changing the score dtype.
        scores = (q @ k_transposed) / np.sqrt(np.float32(self.head_dim))
        ##结束
        
        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        ##我的代码开始
        ## Apply the attention coefficients to the value vectors.
        context = coefficients @ v
        ## Restore token-first order before joining all heads.
        merged = context.transpose(0,2,1,3)
        merged = merged.reshape(x.shape[0],x.shape[1],self.width)
        ##结束

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        ##我的代码开始
        ## The stored output weight is [out, in], so transpose it here.
        output = merged @ self.out_weight.T + self.out_bias
        return output
        ##结束


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        ##我的代码开始
        ## Keep the original input through the first residual connection.
        residual = x + update
        ##结束
        
        # Given: the second Norm and the exact checkpoint activation helper.
        
        ##我的代码开始
        ## Normalize the residual before the MLP sublayer.
        u = self.norm2(residual)
        ##结束

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        
        ##我的代码开始
        ## Expand the channel dimension for the MLP hidden layer.
        hidden  = u @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)
        ## Apply GELU, project back to 768 features, and add the residual.
        MLP = activated @ self.fc2_weight.T + self.fc2_bias
        return residual + MLP
        ##结束

class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        
        ##我的代码开始
        B,C,H,W = images.shape
        ## Expose the 7-by-7 grid and 32-by-32 pixels of each patch.
        patches = images.reshape(B,C,7,32,7,32)
        ## Place patch row/column before channels and local pixels.
        patches = patches.transpose(0,2,4,1,3,5)
        ## Flatten each patch while preserving row-major patch order.
        patches = patches.reshape(B,49,3072)
        return patches
        ##结束
    

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        ##我的代码开始
        ## Map each 3072-value patch to a 768-dimensional token.
        patche_tokens = patches @ self.patch_weight.T
        ##结束

        # Given: one supplied CLS row per image.
        
        ##我的代码开始
        ## Make one CLS token for each image in the batch.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))
        ## Insert CLS at position zero, ahead of all patch tokens.
        tokens = np.concatenate([cls,patche_tokens],axis=1)
        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        ## Broadcast learned positional embeddings over the batch.
        tokens += self.position
        return tokens
        ##结束

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        ##我的代码开始
        ## Obtain the full token sequence after the transformer blocks.
        tokens = self.encode_tokens(images)
        

        # Select CLS, apply post_norm, and project with self.projection.
        ## Select only the first (CLS) token from each image.
        cls_tokens = tokens[:,0,:]
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        ## Normalize CLS before the final linear projection.
        normalized = self.post_norm(cls_tokens)
        ## visual.proj is already [in, out]; do not transpose it.
        output = normalized @ self.projection
        ##结束

        return unit_vector(output)
