###################################
# decompse Angular expanded feature map after every ZerConv layer
###################################
import os
from keras.layers.core import *
from keras.layers import Input
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
import numpy as np
#from sklearn.preprocessing import normalize
from keras.models import Model
from keras.optimizers import SGD, Adam

class Graph_expand(Layer):

    def __init__(self,
                 angular_axis=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Graph_expand, self).__init__(**kwargs)
        self.angular_axis = angular_axis

    def build(self, input_shape):
        super(Graph_expand, self).build(input_shape)

    def call(self, x):
        x_features = x[0] #N_f (BatchSize, N, d) or (BatchSize, N, numofrays, d)
        x_graph = x[1]  # N_disk_id (BatchSize, N, cut_num)
        x_graph = K.tf.squeeze(x_graph) # (N, cut_num), only support Batchsize = 1
        x_disk_features = K.tf.gather(x_features, x_graph, axis = 1)  # N_disk_f (BatchSize, N, cut_num, d) or (BatchSize, N, cut_num, numofrays, d)
        if self.angular_axis:
            x_disk_features = K.tf.transpose(x_disk_features,
                                             perm=[0, 1, 3, 2, 4])  # (BatchSize, N, numofrays, cut_num, d)
        return x_disk_features

    def compute_output_shape(self, input_shape):
        in_features_shape = input_shape[0]
        in_graph_shape = input_shape[1]
        if self.angular_axis:
            output_shape = (in_features_shape[0], in_features_shape[1], in_features_shape[-2], in_graph_shape[-1], in_features_shape[-1])
        else:
            output_shape = (in_features_shape[0], in_features_shape[1], in_graph_shape[-1], in_features_shape[-1])
        return output_shape

class Normalized_Graph_expand(Layer):

    def __init__(self,
                 cut_num,
                 angular_axis=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Normalized_Graph_expand, self).__init__(**kwargs)

        self.cut_num = cut_num  # num of discretized points in a patch
        self.angular_axis = angular_axis

    def build(self, input_shape):
        super(Normalized_Graph_expand, self).build(input_shape)

    def call(self, x):
        x_features = x[0]  # N_f (BatchSize, N, d)
        x_graph = x[1]  # N_disk_id (BatchSize, N, cut_num)
        x_graph = K.tf.squeeze(x_graph)  # (N, cut_num), only support Batchsize = 1

        # Normalized N_disk_f by substructing disk-center-f
        tem_x_features = K.tf.expand_dims(x_features, axis=-2)  # (BatchSize, N, 1, d)
        rep_x_feature = K.tf.tile(tem_x_features, [1, 1, self.cut_num, 1])  # (BatchSize, N, cut_num, d)
        x_disk_features = K.tf.gather(x_features, x_graph, axis=1) - rep_x_feature  # N_disk_f (BatchSize, N, cut_num, d)
        if self.angular_axis:
            x_disk_features = K.tf.transpose(x_disk_features,
                                             perm=[0, 1, 3, 2, 4])  # (BatchSize, N, numofrays, cut_num, d)
        return x_disk_features

    def compute_output_shape(self, input_shape):
        in_features_shape = input_shape[0]
        in_graph_shape = input_shape[1]
        if self.angular_axis:
            output_shape = (in_features_shape[0], in_features_shape[1], in_features_shape[-2], in_graph_shape[-1], in_features_shape[-1])
        else:
            output_shape = (in_features_shape[0], in_features_shape[1], in_graph_shape[-1], in_features_shape[-1])
        return output_shape

class resampleGraph_expand(Layer):

    def __init__(self,
                 angular_axis=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(resampleGraph_expand, self).__init__(**kwargs)
        self.angular_axis=angular_axis

    def build(self, input_shape):
        super(resampleGraph_expand, self).build(input_shape)

    def call(self, x):
        x_features = x[0] #X_f (BatchSize, N, d) or (BatchSize, N, numofrays, d)
        x_graph = K.tf.squeeze(x[1])  # X_disk_Nid (N, cut_num), only support Batchsize = 1

        F = K.tf.squeeze(x[2])  # mesh_faces, [num_faces,3]
        # resample_graph_I: corresponding face_id on the original mesh of the sampled points
        I = K.tf.squeeze(x[3])  # [sampled_N]
        # resample_graph_I: interploation weights of the three vertices in each corresponding face
        B = K.tf.expand_dims(x[4], axis=-1)  # [BatchSize, sampled_N, 3, 1]

        mapping_face_Xid = K.tf.squeeze(K.tf.gather(F,I,axis=0)) # [sampled_N, 3]
        mapping_face_Xfeature = K.tf.gather(x_features, mapping_face_Xid, axis=1) # [BatchSize, sampled_N, 3, d] or [BatchSize, sampled_N, 3, numofrays, d]

        if self.angular_axis:
            feature_channels = x_features.shape[-1]
            angular_channels = x_features.shape[-2]
            B = K.tf.expand_dims(B, axis=-2)
            rep_B = K.tf.tile(B, [1, 1, 1, angular_channels,feature_channels])  # [BatchSize, sampled_N, 3,numofrays,d]
        else:
            feature_channels = x_features.shape[-1]
            rep_B = K.tf.tile(B, [1, 1, 1, feature_channels])  # [BatchSize, sampled_N, 3, d]

        N_x_features = K.tf.reduce_sum(mapping_face_Xfeature*rep_B, axis=2) # N_X_f (BatchSize, sampled_N, numofrays, d), only support Batchsize = 1
        x_disk_features = K.tf.gather(N_x_features, x_graph, axis = 1) # N_disk_f (BatchSize, N, cut_num, numofrays, d)

        if self.angular_axis:
            x_disk_features = K.tf.transpose(x_disk_features, perm = [0,1,3,2,4]) #(BatchSize, N, numofrays, cut_num, d)

        return x_disk_features

    def compute_output_shape(self, input_shape):
        in_features_shape = input_shape[0]
        in_graph_shape = input_shape[1]
        if self.angular_axis:
            output_shape = (in_features_shape[0], in_features_shape[1], in_features_shape[-2], in_graph_shape[-1], in_features_shape[-1])
        else:
            output_shape = (in_features_shape[0], in_features_shape[1], in_graph_shape[-1], in_features_shape[-1])
        return output_shape

class Normalized_resampleGraph_expand(Layer):

    def __init__(self,
                 cut_num,
                 angular_axis=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Normalized_resampleGraph_expand, self).__init__(**kwargs)

        self.cut_num = cut_num     # num of discretized points in a patch
        self.angular_axis=angular_axis

    def build(self, input_shape):
        super(Normalized_resampleGraph_expand, self).build(input_shape)

    def call(self, x):
        x_features = x[0] #X_f (BatchSize, N, d) or (BatchSize, N, numofrays, d)
        x_graph = K.tf.squeeze(x[1])  # X_disk_Nid (N, cut_num), only support Batchsize = 1

        F = K.tf.squeeze(x[2])  # mesh_faces, [num_faces,3]
        # resample_graph_I: corresponding face_id on the original mesh of the sampled points
        I = K.tf.squeeze(x[3])  # [sampled_N]
        # resample_graph_I: interploation weights of the three vertices in each corresponding face
        B = K.tf.expand_dims(x[4], axis=-1)  # [BatchSize, sampled_N, 3, 1]

        mapping_face_Xid = K.tf.squeeze(K.tf.gather(F,I,axis=0)) # [sampled_N, 3]
        mapping_face_Xfeature = K.tf.gather(x_features, mapping_face_Xid, axis=1) # [BatchSize, sampled_N, 3, d] or [BatchSize, sampled_N, 3, numofrays, d]

        if self.angular_axis:
            feature_channels = x_features.shape[-1]
            angular_channels = x_features.shape[-2]
            B = K.tf.expand_dims(B, axis=-2)
            rep_B = K.tf.tile(B, [1, 1, 1,angular_channels,feature_channels])  # [BatchSize, sampled_N, 3,numofrays,d]
            x_features = K.tf.expand_dims(x_features, axis=-3)  # (BatchSize, N, 1, numofrays, d)
            rep_x_feature = K.tf.tile(x_features, [1, 1, self.cut_num, 1, 1])  # (BatchSize, N, cut_num, numofrays, d)
        else:
            feature_channels = x_features.shape[-1]
            rep_B = K.tf.tile(B, [1, 1, 1, feature_channels])  # [BatchSize, sampled_N, 3, d]

            x_features = K.tf.expand_dims(x_features, axis=-2)  # (BatchSize, N, 1, d)
            rep_x_feature = K.tf.tile(x_features, [1, 1, self.cut_num, 1])  # (BatchSize, N, cut_num, d)

        N_x_features = K.tf.reduce_sum(mapping_face_Xfeature*rep_B, axis=2) # N_X_f (BatchSize, sampled_N, numofrays, d), only support Batchsize = 1
        x_disk_features = K.tf.gather(N_x_features, x_graph, axis = 1) - rep_x_feature # N_disk_f (BatchSize, N, cut_num, numofrays, d)

        if self.angular_axis:
            x_disk_features = K.tf.transpose(x_disk_features, perm = [0,1,3,2,4]) #(BatchSize, N, numofrays, cut_num, d)

        return x_disk_features

    def compute_output_shape(self, input_shape):
        in_features_shape = input_shape[0]
        in_graph_shape = input_shape[1]
        if self.angular_axis:
            output_shape = (in_features_shape[0], in_features_shape[1], in_features_shape[-2], in_graph_shape[-1], in_features_shape[-1])
        else:
            output_shape = (in_features_shape[0], in_features_shape[1], in_graph_shape[-1], in_features_shape[-1])
        return output_shape

class ZernikeDecomp(Layer):

    def __init__(self,
                 angular_axis=False,
                 numofrays = 1,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(ZernikeDecomp, self).__init__(**kwargs)

        self.angular_axis = angular_axis
        self.numofrays = numofrays

    def build(self, input_shape):
        super(ZernikeDecomp, self).build(input_shape)

    def call(self, x):
        x_disk_features = x[0] # X_disk_f (BatchSize, N, cut_num, d) or (BatchSize, N, numofrays, cut_num, d)
        x_zerfuncs = x[1] # X_disk_zerfunc (BatchSize, N, cut_num, basis_num);    basis_num = 21 with default zernike_order = 5
        ########################################################################################################################
        if self.angular_axis:
            x_zerfuncs = K.tf.expand_dims(x_zerfuncs, axis=-3)
            x_zerfuncs = K.tf.tile(x_zerfuncs, [1,1,self.numofrays,1,1]) # (BatchSize, N, numofrays, cut_num, basis_num)
        ########################################################################################################################
        x_zernike_coeff = K.tf.matrix_solve_ls(x_zerfuncs,x_disk_features,l2_regularizer=0.01) # X_zernike_coeff (BatchSize, N, basis_num, d) or (BatchSize, N, numofrays, basis_num, d)
        return x_zernike_coeff

    def compute_output_shape(self, input_shape):
        in_features_shape = input_shape[0]
        in_bases_shape = input_shape[1]
        if self.angular_axis:
            output_shape = (in_features_shape[0], in_features_shape[1], in_features_shape[2], in_bases_shape[-1], in_features_shape[-1])
        else:
            output_shape = (in_features_shape[0], in_features_shape[1], in_bases_shape[-1], in_features_shape[-1])
        return output_shape

class ZernikeConv(Layer):

    def __init__(self,
                 filters,
                 zernike_order=5,
                 numofrays=1,
                 angular_axis = False,
                 angular_expand=False,
                 angular_pooling=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):


        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(ZernikeConv, self).__init__(**kwargs)

        self.filters = filters
        self.zernike_order = zernike_order
        self.numofrays = numofrays
        self.angular_axis = angular_axis
        self.angular_expand = angular_expand
        self.angular_pooling = angular_pooling
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        if angular_axis:
            self.angular_expand = False
        else:
            self.angular_pooling = False

    def build(self, input_shape):
        #input_shape: X_zernike_coeff (BatchSize, N, basis_num, d) or (BatchSize, N, numofrays, basis_num, d)
        input_channels = input_shape[-1] # d
        kernel_shape = (input_shape[-2], input_channels, self.filters)   #(21,d,nf)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, x):
        #x: X_zernike_coeff (BatchSize, N, basis_num, d) or (BatchSize, N, numofrays, basis_num, d)
        if self.angular_expand:
            # Angular expansion
            kernel_angular_expand = filter_expand(self.kernel,self.zernike_order,
                                                  self.numofrays) # tensor shape (numofrays, basis_num, d, nf)
            x_conv = K.tf.tensordot(x, kernel_angular_expand,
                                    axes=[[-2, -1], [1, 2]])  # tensor shape (BatchSize, N, numofrays, nf)
            if self.use_bias:
                x_conv += K.reshape(self.bias, (1, 1, 1, self.filters))
        else:
            x_conv = K.tf.tensordot(x, self.kernel, axes=[[-2, -1], [0, 1]])  # tensor shape (BatchSize, N, nf)
            if self.use_bias:
                x_conv += K.reshape(self.bias, (1, 1, self.filters))

        x_conv = self.activation(x_conv) # tensor shape (BatchSize, N, nf) or (BatchSize, N, numofrays, nf)

        if self.angular_pooling:
            # max angular pooling
            x_conv = K.max(x_conv, axis=2)  # tensor shape (BatchSize, N, nf)

        return x_conv

    def compute_output_shape(self, input_shape):

        if self.angular_axis:
            if self.angular_pooling:
                output_shape = (input_shape[0], input_shape[1], self.filters)
            else:
                output_shape = (input_shape[0], input_shape[1], input_shape[2], self.filters)
        else:
            if self.angular_expand:
                output_shape = (input_shape[0], input_shape[1], self.numofrays, self.filters)
            else:
                output_shape = (input_shape[0], input_shape[1], self.filters)

        return output_shape


def filter_expand(filter_in,zernike_order,n_rays):
    x_expand_list = []
    x_expand_list.append(filter_in)
    for i in range(1,n_rays):
        rota_angle = i*2*(np.pi)/n_rays
        x_rot = zernike_rota(filter_in,rota_angle,zernike_order)
        x_expand_list.append(x_rot)
    x_expand = K.tf.stack(x_expand_list,axis=0) #expanded filters shape: (n_rays,basis_num,d,nf)
    return x_expand

def zernike_rota(filter_in, rota_angle, zernike_order):   #filter shape: (basis_num,d,nf)
    x_rot_list = K.tf.unstack(K.tf.identity(filter_in), axis=0)
    for i in range(zernike_order):
        n = i + 1
        if n % 2 == 0:
            for m in range(2, n + 1, 2):
                rota_matrix = K.tf.stack([K.tf.cos(m * rota_angle),
                                          K.tf.sin(m * rota_angle),
                                          -K.tf.sin(m * rota_angle),
                                          K.tf.cos(m * rota_angle)])
                rota_matrix = K.tf.cast(K.tf.reshape(rota_matrix, (2, 2)), dtype=K.tf.float32)
                id_1 = int(n * (n + 1) / 2 + (m + n) / 2)
                id_2 = int(n * (n + 1) / 2 + (-m + n) / 2)

                x_slices = K.tf.gather(filter_in, [id_1, id_2], axis=0) # shape (2, d, nf)
                x_rot_slices = K.tf.tensordot(rota_matrix, x_slices, axes=[[1], [0]])  # shape (2,d,nf)
                slices_list = K.tf.unstack(x_rot_slices)
                x_rot_list[id_1] = slices_list[0]
                x_rot_list[id_2] = slices_list[1]
        else:
            for m in range(1, n + 1, 2):
                rota_matrix = K.tf.stack([K.tf.cos(m * rota_angle),
                                          K.tf.sin(m * rota_angle),
                                          -K.tf.sin(m * rota_angle),
                                          K.tf.cos(m * rota_angle)])
                rota_matrix = K.tf.cast(K.tf.reshape(rota_matrix, (2, 2)), dtype=K.tf.float32)
                id_1 = int(n * (n + 1) / 2 + (m + n) / 2)
                id_2 = int(n * (n + 1) / 2 + (-m + n) / 2)

                x_slices = K.tf.gather(filter_in, [id_1, id_2], axis=0)
                x_rot_slices = K.tf.tensordot(rota_matrix, x_slices, axes=[[1], [0]])  # shape (2,d,nf)
                slices_list = K.tf.unstack(x_rot_slices)
                x_rot_list[id_1] = slices_list[0]
                x_rot_list[id_2] = slices_list[1]

    filter_rot = K.tf.stack(x_rot_list, axis=0)
    return filter_rot