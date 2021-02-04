import paddle.fluid as fluid

from paddle.fluid.dygraph import Conv2D, Conv3D, to_variable, Linear

import numpy as np

def my_conv2d(name=None, x=None, filterdata = None, padding=0, stride =1):
    k_param_attrs = fluid.ParamAttr(
                                initializer=fluid.initializer.NumpyArrayInitializer(filterdata),
                                trainable=True)
    
    conv2d_grad_x = Conv2D(num_channels=x.shape[1], num_filters=1, filter_size=filterdata.shape[3], stride=stride, padding=padding, param_attr=k_param_attrs)
    conv2dx = conv2d_grad_x(x)

    return conv2dx


class FlowLayer(fluid.dygraph.Layer):

    def __init__(self, channels=1, bottleneck=32, params=[1,1,1,1,1], n_iter=20):
        super(FlowLayer, self).__init__()
        
        #self.bottleneck = Conv3D(num_channels=channels, num_filters=bottleneck, filter_size=1, stride=1, padding=0, bias_attr=False)
        self.bottleneck = Conv2D(num_channels=channels, num_filters=bottleneck, filter_size=1, stride=1, padding=0, bias_attr=False)
        #self.unbottleneck = Conv3D(num_channels=bottleneck*2, num_filters=channels, filter_size=1, stride=1, padding=0, bias_attr=False)
        self.unbottleneck = Conv2D(num_channels=bottleneck*2, num_filters=channels, filter_size=1, stride=1, padding=0, bias_attr=False)
        self.bn = fluid.BatchNorm(channels)
        channels = bottleneck

        
        self.n_iter = n_iter
        if params[0]:
            #self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).repeat(channels,channels,1,1))
            self.img_grad = np.array([[[[-0.5,0,0.5]]]*channels]*channels)   #fluid.dygraph.to_variable(np.array([[[[-0.5,0,0.5]]]*channels]*channels))
            self.img_grad_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.img_grad.shape[3], bias_attr=False,
                                            padding = (0, 1),
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.img_grad),
                                                                    trainable=True))
            
            self.img_grad2 = np.array([[[[-0.5],[0],[0.5]]]*channels]*channels)  #fluid.dygraph.to_variable(np.array([[[[-0.5],[0],[0.5]]]*channels]*channels))
            self.img_grad2_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.img_grad2.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.img_grad2),
                                                                    trainable=True))
            
        else:
            #self.img_grad = paddle.fluid.create_lod_tensor(np.array([[[[-0.5,0,0.5]]]*channels]*channels), fluid.CPUPlace())
            self.img_grad = np.array([[[[-0.5,0,0.5]]]*channels]*channels)
            self.img_grad_conv2d0 = Conv2D(num_channels = 32, num_filters=1, filter_size=self.img_grad.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.img_grad),
                                                                    trainable=False))
            #self.img_grad2 = paddle.fluid.create_lod_tensor(np.array([[[[-0.5],[0],[0.5]]]*channels]*channels)) 
            self.img_grad2 = np.array([[[[-0.5],[0],[0.5]]]*channels]*channels)  
            self.img_grad2_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.img_grad2.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.img_grad2),
                                                                    trainable=False))      
        if params[1]:       
            #self.f_grad = paddle.fluid.create_lod_tensor(np.array([[[[-1],[1]]]*channels]*channels), fluid.CPUPlace())
            self.f_grad = np.array([[[[-1,1]]]*channels]*channels)
            self.f_grad_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.f_grad.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.f_grad),
                                                                    trainable=True))
            #self.f_grad2 = paddle.fluid.create_lod_tensor(np.array([[[[-1],[1]]]*channels]*channels), fluid.CPUPlace())
            self.f_grad2 = np.array([[[[-1],[1]]]*channels]*channels)
            self.f_grad2_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.f_grad2.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.f_grad2),
                                                                    trainable=True))
            #self.div = paddle.fluid.create_lod_tensor(np.array([[[[-1],[1]]]*channels]*channels), fluid.CPUPlace())
            self.div = np.array([[[[-1,1]]]*channels]*channels)
            self.div_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.div.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.div),
                                                                    trainable=True))
            #self.div2 = paddle.fluid.create_lod_tensor(np.array([[[[-1],[1]]]*channels]*channels), fluid.CPUPlace())
            self.div2 = np.array([[[[-1],[1]]]*channels]*channels)
            self.div2_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.div2.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.div2),
                                                                    trainable=True))
        else:
            #self.f_grad = paddle.fluid.create_lod_tensor(np.array([[[[-1],[1]]]*channels]*channels), fluid.CPUPlace())
            self.f_grad = np.array([[[[-1],[1]]]*channels]*channels)
            self.f_grad_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.f_grad.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.f_grad),
                                                                    trainable=False))
            #self.f_grad2 = paddle.fluid.create_lod_tensor(np.array([[[[-1],[1]]]*channels]*channels), fluid.CPUPlace())
            self.f_grad2 = np.array([[[[-1],[1]]]*channels]*channels)
            self.f_grad2_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.f_grad2.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.f_grad2),
                                                                    trainable=False))
            #self.div = paddle.fluid.create_lod_tensor(np.array([[[[-1],[1]]]*channels]*channels), fluid.CPUPlace())
            self.div = np.array([[[[-1],[1]]]*channels]*channels)
            self.div_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.div.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.div),
                                                                    trainable=False))
            #self.div2 = paddle.fluid.create_lod_tensor(np.array([[[[-1],[1]]]*channels]*channels), fluid.CPUPlace())
            self.div2 = np.array([[[[-1],[1]]]*channels]*channels)
            self.div2_conv2d = Conv2D(num_channels = 32, num_filters=1, filter_size=self.div2.shape[3], bias_attr=False,
                                            param_attr=fluid.ParamAttr(
                                                                    initializer=fluid.initializer.NumpyArrayInitializer(self.div2),
                                                                    trainable=False))
        self.channels = channels
        
        self.t = np.array([0.3])
        self.l = np.array([0.15])
        self.a = np.array([0.25])
        #self.a = to_variable(self.a)        

        if params[2]:
            #self.t = nn.Parameter(torch.FloatTensor([self.t]))
            #self.t = to_variable(np.array([self.t])) #fluid.Tensor()
            self.t_linear = Linear(1,1, bias_attr=False,
                                    param_attr=fluid.ParamAttr(
                                        initializer=fluid.initializer.NumpyArrayInitializer(self.t),
                                        trainable=True))
            #print('-------------', self.t.stop_gradient)
            #self.t.set(np.array([0.3]), fluid.CUDAPlace(0))
        if params[3]:
            self.l_linear = Linear(1,1, bias_attr=False,
                                    param_attr=fluid.ParamAttr(
                                        initializer=fluid.initializer.NumpyArrayInitializer(self.l),
                                        trainable=True))
        if params[4]:
            self.a_linear = Linear(1,1, bias_attr=False,
                                    param_attr=fluid.ParamAttr(
                                        initializer=fluid.initializer.NumpyArrayInitializer(self.a),
                                        trainable=True))


    def norm_img(self, x):

        mx = fluid.layers.reduce_max(x)
        mn = fluid.layers.reduce_min(x)
        x = 255*(x-mn)/(mn-mx+1e-12)
        return x
            
    def forward_grad(self, x):
        grad_x = self.f_grad_conv2d(fluid.layers.pad(x, paddings=[0,0,0,0,0,0,0,1]))  #my_conv2d(name="grad_x",x=fluid.layers.pad(x, paddings=[0,0,0,0,0,1,0,0]), filterdata=self.f_grad)
        temp = grad_x[:,:,:,-1]
        temp = fluid.layers.zeros_like(temp)
        grad_x = grad_x[:,:,:,0:-1]
        ctemp, htemp, wtemp = temp.shape       
        temp = fluid.layers.reshape(temp, shape = [ctemp,htemp,wtemp,1])
        
        grad_x = fluid.layers.concat([grad_x, temp], axis = 3)   #grad_x.numpy()
 
        
        grad_y = self.f_grad2_conv2d(fluid.layers.pad(x, paddings=[0,0,0,0,0,1,0,0]))  #my_conv2d(name="grad_y", x=fluid.layers.pad(x, paddings=[0,0,0,0,0,1,0,0]), filterdata=self.f_grad2)
        temp = grad_y[:,:,-1,:]
        temp = fluid.layers.zeros_like(temp)
        grad_y = grad_y[:,:,0:-1,:]
        ctemp, htemp, wtemp = temp.shape       
        temp = fluid.layers.reshape(temp, shape = [ctemp,htemp,1,wtemp])
        grad_y = fluid.layers.concat([grad_y, temp], axis = 2)
        return grad_x, grad_y


    def divergence(self, x, y):
        tx = fluid.layers.pad(x=x[:,:,:,:-1], paddings=[0,0,0,0,0,0,1,0])
        ty = fluid.layers.pad(x=y[:,:,:-1,:], paddings=[0,0,0,0,1,0,0,0])
        grad_x = self.div_conv2d(fluid.layers.pad(tx, paddings=[0,0,0,0,0,0,0,1]))     #my_conv2d(name="grad_x", x=fluid.layers.pad(tx, paddings=[0,0,0,0,0,1,0,0]), filterdata=self.div,  padding=0, stride=1)

        grad_y = self.div2_conv2d(fluid.layers.pad(ty, paddings=[0,0,0,0,0,1,0,0]))    #my_conv2d(name="grad_y", x=fluid.layers.pad(ty, paddings=[0,0,0,0,0,1,0,0]), filterdata=self.div2)
        return grad_x + grad_y
        
        
    def forward(self, x):
        frames = x.shape[1]
        residual = x[:,:-1,:]
        x = fluid.layers.reshape(x, shape=[-1, 512, x.shape[3], x.shape[4]])
        x = self.bottleneck(x)
        x = fluid.layers.reshape(x, shape=[-1,frames,32,14,14])
        inp = self.norm_img(x)
        x = inp[:,:-1,:]
        y = inp[:,1:,:]
        b,t,c,h,w = x.shape
        x = fluid.layers.reshape(x, shape=[b*t,c,h,w])
        y = fluid.layers.reshape(y, shape=[b*t,c,h,w])
        
        u1 = fluid.layers.zeros_like(x)
        u2 = fluid.layers.zeros_like(x)
        xone = fluid.layers.ones(shape=[1], dtype='float32')  #to_variable(np.array([1])).astype('float32')
        l_t = self.l_linear(xone) * self.t_linear(xone)
        taut = self.a_linear(xone)/(self.t_linear(xone) + 1e-12)

        grad2_x = self.img_grad_conv2d(y)

        ax = 0.5*(x[:,:,:,1] - x[:,:,:,0])
        bx = 0.5*(x[:,:,:,-1] - x[:,:,:,-2])
        ctemp, htemp, wtemp = ax.shape       
        ax = fluid.layers.reshape(ax, shape = [ctemp,htemp,wtemp,1])
        ctemp, htemp, wtemp = bx.shape    
        bx = fluid.layers.reshape(bx, shape = [ctemp,htemp,wtemp,1])
        grad2_x = grad2_x[:,:,:,1:-1]
        grad2_x = fluid.layers.concat([ax, grad2_x], axis = 3)
        grad2_x = fluid.layers.concat([grad2_x, bx], axis = 3)

        grad2_y = self.img_grad2_conv2d(fluid.layers.pad(x=y, paddings=[0,0,0,0,1,1,0,0])) #my_conv2d(name="grad2_y", x=fluid.layers.pad(x=y, paddings=[0,0,0,0,1,1,0,0]), filterdata=self.img_grad2,padding=0, stride=1)  #Conv2D(fluid.layers.pad(y, (0,0,1,1)), self.img_grad2, padding=0, stride=1)
        
        ax = 0.5*(x[:,:,1,:] - x[:,:,0,:])
        bx = 0.5*(x[:,:,-1,:] - x[:,:,-2,:])
        ctemp, htemp, wtemp = ax.shape       
        ax = fluid.layers.reshape(ax, shape = [ctemp,htemp,1,wtemp])
        ctemp, htemp, wtemp = bx.shape    
        bx = fluid.layers.reshape(bx, shape = [ctemp,htemp,1,wtemp])

        grad2_y = grad2_y[:,:,1:-1,:]
        grad2_y = fluid.layers.concat([ax, grad2_y], axis = 2)
        grad2_y = fluid.layers.concat([grad2_y, bx], axis = 2)

        p11 = fluid.layers.zeros_like(x)
        p12 = fluid.layers.zeros_like(x)
        p21 = fluid.layers.zeros_like(x)
        p22 = fluid.layers.zeros_like(x)

        gsqx = grad2_x**2
        gsqy = grad2_y**2
        grad = gsqx + gsqy + 1e-12

        rho_c = y - grad2_x * u1 - grad2_y * u2 - x
        
        for i in range(self.n_iter):
            rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12

            v1 = fluid.layers.zeros_like(x)
            v2 = fluid.layers.zeros_like(x)

            multemp = l_t*grad

            ltgrad2x = (l_t * grad2_x)
            ltgrad2y = (l_t * grad2_y)

            mask1temp = rho < (-multemp)
            mask1 = mask1temp.astype('float32').detach() 
            v1 = mask1*ltgrad2x
            v2 = mask1*ltgrad2y

            mask2temp = (rho > multemp)
            mask2 = mask2temp.astype('float32').detach()
            v1 = -ltgrad2x*mask2 + v1
            v2 = -ltgrad2y*mask2 + v2

            maskones = fluid.layers.ones_like(mask1)
            mask3 = maskones - mask1 - mask2  #to_variable(mask3).astype('float32').detach()
            v1 = ((-rho/grad) * grad2_x)*mask3 + v1
            v2 = ((-rho/grad) * grad2_y)*mask3 + v2

            del rho
            del mask1
            del mask2
            del mask3

            v1 += u1
            v2 += u2
            temp = self.t_linear(xone)

            u1 = v1 + (temp) * self.divergence(p11, p12)
            u2 = v2 + (temp) * self.divergence(p21, p22)
            del v1
            del v2
            u1 = u1
            u2 = u2

            u1x, u1y = self.forward_grad(u1)
            u2x, u2y = self.forward_grad(u2)

            p11 = (p11 + taut * u1x) / (1. + taut * fluid.layers.sqrt(u1x**2 + u1y**2 + 1e-12))
            p12 = (p12 + taut * u1y) / (1. + taut * fluid.layers.sqrt(u1x**2 + u1y**2 + 1e-12))
            p21 = (p21 + taut * u2x) / (1. + taut * fluid.layers.sqrt(u2x**2 + u2y**2 + 1e-12))
            p22 = (p22 + taut * u2y) / (1. + taut * fluid.layers.sqrt(u2x**2 + u2y**2 + 1e-12))
            del u1x
            del u1y
            del u2x
            del u2y
            


        flow = fluid.layers.concat([u1,u2], axis=1)

        flow = self.unbottleneck(flow)
        flow = self.bn(flow)
        flow = fluid.layers.reshape(flow, shape=[b,t,512,h,w])
        out = residual + flow

        return fluid.layers.relu(out)