#usr/bin/python3
from . import *
from einops import rearrange
from .embedding import *
from .normal import *



class ScaledDotProductAttention(nn.Module):
        
    def __init__(self,dropout=0.0):
        """Calculate the dot product attention from given Q, K, V:
        $$
        \begin{align}
        A\left( \mathbf{Q}, \mathbf{K}, \mathbf{V} \right)=softmax(\frac{\mathbf{Q}\cdot\mathbf{K^\top}}{d_k})\cdot \mathbf{V},
        \\
        \mathbf Q\in\mathbb R^{n\times d_k}, \space \mathbf K\in\mathbb R^{m\times d_k}
        ,\space \mathbf V\in\mathbb R^{m\times d_q}
        \end{align}
        $$

        Args:
            dropout (float, optional): The parametr used to control the dropout of this layer. Defaults to 0.0.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        """Calculate the dot product attention from given Q, K, V.

        Args:
            queries (Tensor): Q tensor. The shape of Q tensor should be (batch_size, n, d_k)
            keys (Tensor): K tensor. The shape of K tensor should be (batch_size, m, d_k)
            values (Tensor): V tensor. The shape of V tensor should be (batch_size, m, d_v)

        Returns:
            Tensor: $softmax(\frac{\mathbf{Q}\cdot\mathbf{K^\top}}{d_k})\cdot \mathbf{V}$ with a shape of (batch_size, n, d_v)
        """
        d_k = keys.shape[-1]
        weights = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d_k)
        weights = nn.functional.softmax(weights, dim=-1)
        return torch.bmm(self.dropout(weights), values)

class LinearScaledDotProductAttention(nn.Module):
        
    def __init__(self,dropout=0.0):
        """Calculate the linear dot product attention from given Q, K, V:
        $$
        \begin{align}
                A\left( \mathbf{Q}, \mathbf{K}, \mathbf{V} \right)
                =softmax_{row}(\mathbf{Q}）\left[\cdot softmax_{column}(\mathbf{K^\top})\cdot \mathbf{V}\right],
                \\
                \mathbf Q\in\mathbb R^{n\times d_k}, \space \mathbf K\in\mathbb R^{m\times d_k}
                ,\space \mathbf V\in\mathbb R^{m\times d_q}
        \end{align}
        $$
        see more at ["Efficient Attention: Attention with Linear Complexities"](http://arxiv.org/abs/1812.01243)

        Args:
            dropout (float, optional): The parametr used to control the dropout of this layer. Defaults to 0.0.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        """Calculate the linear dot product attention from given Q, K, V.

        Args:
            queries (Tensor): Q tensor. The shape of Q tensor should be (batch_size, n, d_k)
            keys (Tensor): K tensor. The shape of K tensor should be (batch_size, m, d_k)
            values (Tensor): V tensor. The shape of V tensor should be (batch_size, m, d_v)

        Returns:
            Tensor: $softmax_{row}(\mathbf{Q}）\left[\cdot softmax_{column}(\mathbf{K^\top})\cdot \mathbf{V}\right]$ with a shape of (batch_size, n, d_v)
        """
        weights=(nn.functional.softmax(keys,dim=-2)).transpose(1,2)
        weights=torch.bmm(weights,values)
        return torch.bmm(nn.functional.softmax(queries,dim=-1),self.dropout(weights))

class MultiHeadAttentionBase(nn.Module):
    
    def __init__(self, num_heads:int,linear_attention=False, dropout=0.0):
        """Base class for multi-head attention. The inut of this class (i.e. Q,K,V) 
        should be (batch_size, num_elements, num_heads$\times$dim_deads).

        Args:
            num_heads (int): The number of attention heads.
            linear_attention (bool, optional): Whether to use linear attention. Defaults to False.
            dropout (float, optional): The parametr used to control the dropout of this layer. Defaults to 0.0.
        """
        super().__init__()
        self.num_heads = num_heads
        if linear_attention:
            self.attention=LinearScaledDotProductAttention(dropout=dropout)
        else:
            self.attention = ScaledDotProductAttention(dropout=dropout)


    def forward(self, queries, keys, values):
        """Calculate the multi-head dot product attention from given Q, K, V.

        Args:
            queries (Tensor): Q tensor. The shape of Q tensor should be (batch_size, n, num_heads$\times$dim_deads)
            keys (Tensor): K tensor. The shape of K tensor should be (batch_size, m, num_heads$\times$dim_deads)
            values (Tensor): V tensor. The shape of V tensor should be (batch_size, m, num_heads$\times$dim_deads)

        Returns:
            Tensor: Multi-head attention result with a shape of (batch_size, n, num_heads$\times$dim_deads)
        """
        queries,keys,values =map(self.apart_input,(queries,keys,values))
        output = self.attention(queries, keys, values)
        return self.concat_output(output)


    def apart_input(self,x):
        """Transform the input tensor with shape of (batch_size, num_elements, num_heads$\times$dim_deads) to the
        output tensor with the shape of (batch_size$\times$num_heads, num_elements, dim_deads). The aim is to accelerate
        the computation with the batch dot product (torch.bmm) operation in ScaledDotProductAttention. This is the 
        inverse operation of concat_output(). The code is from the book ["Dive into deep learning"](https://github.com/d2l-ai/)
        
        Args:
            x (Tensor): Input tensor with the shape of (batch_size, num_elements, num_heads$\times$dim_deads).

        Returns:
            Tensor: Output tensor with the shape of (batch_size$\times$num_heads, num_elements, dim_deads).
        """
        #(batch_size, num_elements, num_heads$\times$dim_deads)  >>> (batch_size, num_elements, num_heads, dim_deads)
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        #(batch_size, num_elements, num_heads, dim_deads) >>> (batch_size, num_heads, num_elements, dim_deads) 
        x = x.permute(0, 2, 1, 3)
        #(batch_size, num_heads, num_elements, dim_deads)  >>> (batch_size$\times$num_heads, num_elements, dim_deads) 
        return x.reshape(-1, x.shape[2], x.shape[3])


    def concat_output(self, x):
        """Transform the input tensor with shape of (batch_size$\times$num_heads, num_elements, dim_deads) to the
        output tensor with shape of (batch_size, num_elements, num_heads$\times$dim_deads). The aim is to accelerate
        the computation with the batch dot product (torch.bmm) operation in ScaledDotProductAttention. This is the 
        inverse operation of apart_qkv(). The code is from the book ["Dive into deep learning"](https://github.com/d2l-ai/)

        Args:
            x (Tensor): Input tensor with the shape of (batch_size$\times$num_heads, num_elements, dim_deads).

        Returns:
            Tensor: Output tensor with the shape of (batch_size, num_elements, num_heads$\times$dim_deads).
        """
        #(batch_size$\times$num_heads, num_elements, dim_deads) >>> (batch_size, num_heads, num_elements, dim_deads)
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        #(batch_size, num_heads, num_elements, dim_deads) >>> (batch_size, num_elements, num_heads, dim_deads)
        x = x.permute(0, 2, 1, 3)
        #(batch_size, num_elements, num_heads, dim_deads) >>> (batch_size, num_elements, num_heads$\times$dim_deads)
        return x.reshape(x.shape[0], x.shape[1], -1)

class SequenceMultiHeadAttention(nn.Module):
    def __init__(self,dim_q:int, dim_k:int, dim_v:int, num_heads:int, dim_heads:int,dim_out:int, linear_attention=False, dropout=0.0,bias=False):
        """Calculate the multihead attention for a sequence data. The shape of inputs Q, K, V should
        be (batch_size, num_sequence_elements, dim_elements). The attention calculation is performed on each sequence elements.

        Args:
            dim_q (int): The dimension of Q tensor.
            dim_k (int): The dimension of K tensor.
            dim_v (int): The dimension of V tensor.
            num_heads (int): The number of attention heads.
            dim_heads (int): The dimension of attention heads.
            dim_out (int): The dimension of output tensor.
            linear_attention (bool, optional): Whether to use linear attention. Defaults to False.
            dropout (float, optional): The parametr used to control the dropout of this layer. Defaults to 0.0.
            bias (bool, optional): Whether to use bias when projecting the inputs and outputs. Defaults to False.
        """
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_q = nn.Linear(dim_q, dim_hiddens,bias=bias)
        self.w_k = nn.Linear(dim_k, dim_hiddens,bias=bias)
        self.w_v = nn.Linear(dim_v, dim_hiddens,bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Linear(dim_hiddens, dim_out,bias=bias)
    
    def forward(self, queries, keys, values):
        """Calculate the multi-head dot product attention from given Q, K, V. 
        The shape of Q,K,V tensor should be (batch_size, num_sequence_elements, dim_elements).
        Since Q,K,V will all be projected into a new space with the channel of $num_heads*dim_heads$,
        there is no need to guaranteen a same dim_elements of Q tensor and K tensor.
        The only limit is that the num_sequence_elements of K and V should be the same.

        Args:
            queries (Tensor): Q tensor. 
            keys (Tensor): K tensor.
            values (Tensor): V tensor.

        Returns:
            Tensor: Multi-head attention result with a shape of (batch_size, num_sequence_elements_Q, dim_out).
        """
        q=self.w_q(queries)
        k=self.w_k(keys)
        v=self.w_v(values)
        att=self.mha(q,k,v)
        return self.w_o(att)

class TwoDFieldMultiHeadAttention(nn.Module):

    def __init__(self,dim_q, dim_k, dim_v, num_heads, dim_heads,dim_out, linear_attention=False, dropout=0.0,bias=False):
        """Calculate the multihead attention for a 2D Field data. The shape of inputs Q, K, V should
        be (batch_size, dim_elements, height_field, width_field). The attention calculation is performed on each field elements.

        Args:
            dim_q (int): The dimension of Q tensor.
            dim_k (int): The dimension of K tensor.
            dim_v (int): The dimension of V tensor.
            num_heads (int): The number of attention heads.
            dim_heads (int): The dimension of attention heads.
            dim_out (int): The dimension of output tensor.
            linear_attention (bool, optional): Whether to use linear attention. Defaults to False.
            dropout (float, optional): The parametr used to control the dropout of this layer. Defaults to 0.0.
            bias (bool, optional): Whether to use bias when projecting the inputs and outputs. Defaults to False.
        """
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_q = nn.Conv2d(dim_q, dim_hiddens, 1, bias=bias)
        self.w_k = nn.Conv2d(dim_k, dim_hiddens, 1, bias=bias)
        self.w_v = nn.Conv2d(dim_v, dim_hiddens, 1, bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Conv2d(dim_hiddens, dim_out,1,bias=bias)
    
    def forward(self, queries, keys, values):
        """Calculate the multi-head dot product attention from given Q, K, V. 
        The shape of Q,K,V tensor should be (batch_size, dim_elements, height_field, width_field).
        Since Q,K,V will all be projected into a new space with the channel of $num_heads*dim_heads$,
        there is no need to guaranteen a same dim_elements of Q tensor and K tensor.
        The only limit is that the height_field$\times$width_field of K and V should be the same.

        Args:
            queries (Tensor): Q tensor. 
            keys (Tensor): K tensor.
            values (Tensor): V tensor.

        Returns:
            Tensor: Multi-head attention result with a shape of (batch_size, dim_out, height_field_Q, width_field_Q).
        """
        width=queries.shape[-1]
        q=self.w_q(queries)
        k=self.w_k(keys)
        v=self.w_v(values)
        q, k, v = map(lambda t: rearrange(t, "b c h w -> b (h w) c"), (q,k,v))
        att=self.mha(q,k,v)
        att_2D=rearrange(att,"b (h w) c -> b c h w",w=width)
        return self.w_o(att_2D)

class TwoDFieldMultiHeadSelfAttention(nn.Module):

    def __init__(self,dim_in:int, num_heads:int, dim_heads:int,dim_out:int, linear_attention=False, dropout=0.0,bias=False):
        """Calculate the multihead self-attention for a 2D Field data. The shape of inputs field should
        be (batch_size, dim_elements, height_field, width_field). The attention calculation is performed on each field elements.
        You can also achieve same result using “TwoDFieldMultiHeadAttention” class with same inputs for Q, K and V.

        Args:
            dim_in (int): The dimension of the inut tensor.
            num_heads (int): The number of attention heads.
            dim_heads (int): The dimension of attention heads.
            dim_out (int): The dimension of output tensor.
            linear_attention (bool, optional): Whether to use linear attention. Defaults to False.
            dropout (float, optional): The parametr used to control the dropout of this layer. Defaults to 0.0.
            bias (bool, optional): Whether to use bias when projecting the inputs and outputs. Defaults to False.
        """
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_qkv=nn.Conv2d(dim_in, dim_hiddens*3, 1, bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Conv2d(dim_hiddens, dim_out,1,bias=bias)
    
    def forward(self, x):
        """
        Calculate the multihead self-attention for a 2D Field data x.
        The shape of inputs field should be (batch_size, dim_elements, height_field, width_field). 

        Args:
            x (Tensor): input tensor. 

        Returns:
            Tensor: Multi-head self attention result with a shape of (batch_size, dim_out, height_field_x, width_field_x).
        """
        width=x.shape[-1]
        qkv = self.w_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b c h w -> b (h w) c"), qkv)
        att=self.mha(q,k,v)
        att_2D=rearrange(att,"b (h w) c -> b c h w",w=width)
        return self.w_o(att_2D)

class TwoDFieldMultiHeadChannelSelfAttention(nn.Module):

    def __init__(self,num_pixel:int, num_heads:int, dim_heads:int, linear_attention=False, dropout=0.0,bias=False):
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_q = nn.Linear(num_pixel, dim_hiddens,bias=bias)
        self.w_k = nn.Linear(num_pixel, dim_hiddens,bias=bias)
        self.w_v = nn.Linear(num_pixel, dim_hiddens,bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Linear(dim_hiddens, num_pixel,bias=bias)
    
    def forward(self, x):
        h=x.shape[-1]
        x=rearrange(x, "b c h w -> b c (hw)")
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)
        att=self.w_o(self.mha(q,k,v))
        return rearrange(att, "b c (hw) -> b c h w",h=h)

class PositionalEncoding(nn.Module):

    def __init__(self,dim:int,max_elements_num=10000,dropout=0.0):
        """Add a positional encoding to the input tensor.
        The size of input tensor X should be (batch_size, num_elements, dim_elements).
        The ouput of this module is X+P where the elements of P in each batch are:
        $$
        \begin{split}\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}\end{split}
        $$ 

        Args:
            dim (int): The dimensions of input tensor.
            max_elements_num (int, optional): Expected maximum number of elements. The number of elements in the input (i.e. num_elements) tensor is often unknown when constructing the network, so a maximum value needs to be estimated in advance. Defaults to 10000.
            dropout (float, optional): The parametr used to control the dropout of this layer. Defaults to 0.0. Defaults to 0.0.
        """
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        indexs_row = torch.arange(max_elements_num, dtype=torch.float32).reshape(-1, 1)
        indexs_col = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        x = indexs_row / torch.pow(10000, indexs_col)
        self.position=torch.zeros((1,max_elements_num,dim))
        self.position[:, :, 0::2] = torch.sin(x)
        self.position[:, :, 1::2] = torch.cos(x)      
    
    def forward(self,x):
        """Add a positional encoding to the input tensor. 

        Args:
            x (tensor): The input tensor with the shape of (batch_size, num_elements, dim_elements).

        Returns:
            Tensor: Encoded tensor with the shape of (batch_size, num_elements, dim_elements).
        """
        x=x+self.position[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class AttentionBlockBase(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dim_heads: int, dim_condition: int, linear_attention=False, dropout=0.0):
        super().__init__()
        dim_k_v = default(dim_condition, dim_in)
        self.att = TwoDFieldMultiHeadAttention(dim_q=dim_in, dim_k=dim_k_v, dim_v=dim_k_v, num_heads=num_heads,
                                               dim_heads=dim_heads, dim_out=dim_out, linear_attention=linear_attention, dropout=dropout)
        self.residual_conv = self.res_conv = nn.Conv2d(
            dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.normal = GroupNormalX(dim_out)

    def forward(self, x, condition):
        k_v = default(condition, x)
        x = self.dropout(self.att(queries=x, keys=k_v,
                         values=k_v))+self.residual_conv(x)
        return self.normal(x)

class SelfAttentionBlock(AttentionBlockBase):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dim_heads: int, linear_attention=False, dropout=0):
        super().__init__(dim_in, dim_out, num_heads, dim_heads, dim_condition=None,
                         linear_attention=linear_attention, dropout=dropout)

    def forward(self, x):
        return super().forward(x, condition=None)

class CrossAttentionBlock(AttentionBlockBase,ConditionEmbModel):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dim_heads: int, dim_condition: int, linear_attention=False, dropout=0):
        super().__init__(dim_in, dim_out, num_heads, dim_heads,
                         dim_condition, linear_attention, dropout)

class ChannelAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_pixel: int, num_heads: int, dim_heads: int, linear_attention=False, dropout=0.0):
        super().__init__()
        self.att = TwoDFieldMultiHeadChannelSelfAttention(
            num_pixel=num_pixel, num_heads=num_heads, dim_heads=dim_heads, linear_attention=linear_attention, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.normal = GroupNormalX(dim)

    def forward(self, x):
        x = self.dropout(self.att(x))+x
        return self.normal(x)
