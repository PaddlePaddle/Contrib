from paddle import fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Dropout, BatchNorm, Sequential
from paddle.fluid.layers import bmm,image_resize,create_parameter, reduce_max, reshape, transpose,softmax,expand_as
from .resnet import ResNet

class DANet(fluid.dygraph.Layer):
    def __init__(self,name_scope,out_chs=20,in_chs=1024,inter_chs=512):
        super(DANet,self).__init__(name_scope)
        name_scope = self.full_name()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.inter_chs = inter_chs if inter_chs else in_chs


        self.backbone = ResNet(50)
        self.conv5p = Sequential(
            Conv2D(self.in_chs, self.inter_chs, 3, padding=1),
            BatchNorm(self.inter_chs,act='relu'),
        )
        self.conv5c = Sequential(
            Conv2D(self.in_chs, self.inter_chs, 3, padding=1),
            BatchNorm(self.inter_chs,act='relu'),
        )

        self.sp = PAM_module(self.inter_chs)
        self.sc = CAM_module(self.inter_chs)

        self.conv6p = Sequential(
            Conv2D(self.inter_chs, self.inter_chs, 3, padding=1),
            BatchNorm(self.inter_chs,act='relu'),
        )
        self.conv6c = Sequential(
            Conv2D(self.inter_chs, self.inter_chs, 3, padding=1),
            BatchNorm(self.inter_chs,act='relu'),
        )

        self.conv7p = Sequential(
            Dropout(0.1),
            Conv2D(self.inter_chs, self.out_chs, 1),
        )
        self.conv7c = Sequential(
            Dropout(0.1),
            Conv2D(self.inter_chs, self.out_chs, 1),
        )
        self.conv7pc = Sequential(
            Dropout(0.1),
            Conv2D(self.inter_chs, self.out_chs, 1),
        )

    def forward(self,x):

        feature = self.backbone(x)

        p_f = self.conv5p(feature)
        p_f = self.sp(p_f)
        p_f = self.conv6p(p_f)
        p_out = self.conv7p(p_f)

        c_f = self.conv5c(feature)
        c_f = self.sc(c_f)
        c_f = self.conv6c(c_f)
        c_out = self.conv7c(c_f)

        sum_f = p_f+c_f
        sum_out = self.conv7pc(sum_f)

        p_out = image_resize(p_out,out_shape=x.shape[2:])
        c_out = image_resize(c_out,out_shape=x.shape[2:])
        sum_out = image_resize(sum_out,out_shape=x.shape[2:])
        return [p_out, c_out, sum_out]
        # return sum_out

class PAM_module(fluid.dygraph.Layer):
    def __init__(self,in_chs,inter_chs=None):
        super(PAM_module,self).__init__()
        self.in_chs = in_chs
        self.inter_chs = inter_chs if inter_chs else in_chs
        self.conv_query = Conv2D(self.in_chs,self.inter_chs,1)
        self.conv_key = Conv2D(self.in_chs,self.inter_chs,1)
        self.conv_value = Conv2D(self.in_chs,self.inter_chs,1)
        self.gamma = create_parameter([1], dtype='float32')
    
    def forward(self,x):
        b,c,h,w = x.shape

        f_query = self.conv_query(x)
        f_query = reshape(f_query,(b, -1, h*w))
        f_query = transpose(f_query,(0, 2, 1)) 

        f_key = self.conv_key(x)
        f_key = reshape(f_key,(b, -1, h*w))

        f_value = self.conv_value(x)
        f_value = reshape(f_value,(b, -1, h*w))
        f_value = transpose(f_value,(0, 2, 1)) 


        f_similarity = bmm(f_query, f_key)                        # [h*w, h*w]
        f_similarity = softmax(f_similarity)
        f_similarity = transpose(f_similarity,(0, 2, 1))

        f_attention = bmm(f_similarity, f_value)                        # [h*w, c]
        f_attention = reshape(f_attention,(b,c,h,w))

        out = self.gamma*f_attention + x
        return out

class CAM_module(fluid.dygraph.Layer):
    def __init__(self,in_chs,inter_chs=None):
        super(CAM_module,self).__init__()
        self.in_chs = in_chs
        self.inter_chs = inter_chs if inter_chs else in_chs
        self.gamma = create_parameter([1], dtype='float32')

    def forward(self,x):
        b,c,h,w = x.shape

        f_query = reshape(x,(b, -1, h*w))
        f_key = reshape(x,(b, -1, h*w))
        f_key = transpose(f_key,(0, 2, 1)) 
        f_value = reshape(x,(b, -1, h*w))

        f_similarity = bmm(f_query, f_key)                        # [h*w, h*w]
        f_similarity_max = reduce_max(f_similarity, -1, keep_dim=True)
        f_similarity_max_reshape = expand_as(f_similarity_max,f_similarity)
        f_similarity = f_similarity_max_reshape-f_similarity

        f_similarity = softmax(f_similarity)
        f_similarity = transpose(f_similarity,(0, 2, 1)) 

        f_attention = bmm(f_similarity,f_value)                        # [h*w, c]
        f_attention = reshape(f_attention,(b,c,h,w))

        out = self.gamma*f_attention + x
        return out












