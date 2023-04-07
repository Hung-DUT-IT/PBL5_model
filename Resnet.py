import tensorflow as tf
from keras.layers import Conv2D, Activation, Input, MaxPooling2D, Dense, Dropout, BatchNormalization, Concatenate, Lambda, add, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K


def scaling(x, scale):
    return x * scale


def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', use_bias=False, name=None, activation='relu'):
    """
    Utility function to apply Conv2D + Batch normalization + activation.
    """
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,scale=False, name=name + '_BatchNorm')(x)

    x = Activation(activation, name=name + '_Activation')(x)
    return x
# def conv2d_branch()


def Stem(x):
    x = conv2d_bn(x, 32, 3, strides=2, padding='valid', use_bias=False, name='Conv2d_1a_3x3')
    x = conv2d_bn(x, 32, 3, strides=1, padding='valid', use_bias=False, name='Conv2d_2a_3x3')
    x = conv2d_bn(x, 64, 3, strides=1, padding='same',use_bias=False, name='Conv2d_2b_3x3')
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    x = conv2d_bn(x, 80, 1, strides=1, padding='valid', use_bias=False, name='Conv2d_3b_1x1')
    x = conv2d_bn(x, 192, 3, strides=1, padding='valid', use_bias=False, name='Conv2d_4a_3x3')
    x = conv2d_bn(x, 256, 3, strides=2, padding='valid', use_bias=False, name='Conv2d_4b_3x3')
    return x


def inception_ResNet_A(x):

    names = ['Block35_1', 'Block35_2', 'Block35_3', 'Block35_4', 'Block35_5']
    for i in names:
        branch_0 = conv2d_bn(x, 32, 1, strides=1, padding='same', use_bias=False, name=i + '_Branch_0_Conv2d_1x1')

        branch_1 = conv2d_bn(x, 32, 1, strides=1, padding='same',use_bias=False, name=i + '_Branch_1_Conv2d_0a_1x1')
        branch_1 = conv2d_bn(branch_1, 32, 3, strides=1, padding='same', use_bias=False, name=i + '_Branch_1_Conv2d_0b_3x3')

        branch_2 = conv2d_bn(x, 32, 1, strides=1, padding='same', use_bias=False, name=i + '_Branch_2_Conv2d_0a_1x1')
        branch_2 = conv2d_bn(branch_2, 32, 3, strides=1, padding='same', use_bias=False, name=i + '_Branch_2_Conv2d_0b_3x3')
        branch_2 = conv2d_bn(branch_2, 32, 3, strides=1, padding='same', use_bias=False, name=i + '_Branch_2_Conv2d_0c_3x3')

        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name=i + '_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same',use_bias=True, name=i + '_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name=i + '_Activation')(x)
    return x


def reduction_A(x):
    # Mixed 6a (Reduction-A block):
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid', use_bias=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3')

    branch_1 = conv2d_bn(x, 192, 1, strides=1, padding='same', use_bias=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1')
    branch_1 = conv2d_bn(branch_1, 192, 3, strides=1, padding='same', use_bias=False, name='Mixed_6a_Branch_1_Conv2d_0b_3x3')
    branch_1 = conv2d_bn(branch_1, 256, 3, strides=2, padding='valid', use_bias=False, name='Mixed_6a_Branch_1_Conv2d_1a_3x3')

    branch_pool = MaxPooling2D( 3, strides=2, padding='valid', name='Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)

    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=3, name='Mixed_6a')(branches)
    return x


def inception_resNet_B(x):
    # 10x Block17 (Inception-ResNet-B block):
	names = ['Block17_1', 'Block17_2','Block17_3','Block17_4', 'Block17_5','Block17_6', 'Block17_7','Block17_8','Block17_9', 'Block17_10']

	for i in names:
		branch_0 = conv2d_bn(x, 128, 1, strides=1, padding='same', use_bias=False, name= i + '_Branch_0_Conv2d_1x1')

		branch_1 = conv2d_bn(x, 128, 1, strides=1, padding='same', use_bias=False, name= i + '_Branch_1_Conv2d_0a_1x1')
		branch_1 = conv2d_bn(branch_1, 128, [1, 7], strides=1, padding='same', use_bias=False, name= i + '_Branch_1_Conv2d_0b_1x7')
		branch_1 = conv2d_bn(branch_1, 128, [7, 1], strides=1, padding='same', use_bias=False, name= i + '_Branch_1_Conv2d_0c_7x1')

		branches = [branch_0, branch_1]
		mixed = Concatenate(axis=3, name= i + '_Concatenate')(branches)
		up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= i +'_Conv2d_1x1') (mixed)
		up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
		x = add([x, up])
		x = Activation('relu', name= i + '_Activation')(x)

	return x

def reduction_B(x):
	# Mixed 7a (Reduction-B block): 8 x 8 x 2080	
	branch_0 = conv2d_bn(x, 256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_0a_1x1')
	branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_1a_3x3')

	branch_1 = conv2d_bn(x, 256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_0a_1x1')
	branch_1 = conv2d_bn(branch_1, 256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_1a_3x3')
	
	branch_2 = conv2d_bn(x, 256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0a_1x1')
	branch_2 = conv2d_bn(branch_2, 256, 3, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0b_3x3')    
	branch_2 = conv2d_bn(branch_2, 256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_1a_3x3')  
        
	branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
	branches = [branch_0, branch_1, branch_2, branch_pool]
	x = Concatenate(axis=3, name='Mixed_7a')(branches)
	return x

def inception_resNet_C(x):
	# 5x Block8 (Inception-ResNet-C block):

	names = ['Block8_1', 'Block8_2', 'Block8_3', 'Block8_4', 'Block8_5', 'Block8_6']
	for i in names: 
		branch_0 = conv2d_bn(x, 192, 1, strides=1, padding='same', use_bias=False, name= i + '_Branch_0_Conv2d_1x1')
                
		branch_1 = conv2d_bn(x, 192, 1, strides=1, padding='same', use_bias=False, name= i + '_Branch_1_Conv2d_0a_1x1')
		branch_1 = conv2d_bn(branch_1, 192, [1, 3], strides=1, padding='same', use_bias=False, name= i + '_Branch_1_Conv2d_0b_1x3')
		branch_1 = conv2d_bn(branch_1, 192, [3, 1], strides=1, padding='same', use_bias=False, name= i + '_Branch_1_Conv2d_0b_3x1')
		branches = [branch_0, branch_1]
		mixed = Concatenate(axis=3, name= i + '_Concatenate')(branches)
		up = Conv2D(1792, 1, strides=1, padding='same',use_bias=True, name= i + '_Conv2d_1x1')(mixed)
		up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
		x = add([x, up])
		x = Activation('relu', name= i + '_Activation')(x)    

	return x


def InceptionResNetV2():
    inputs = Input(shape=(160, 160, 3))
    x = Stem(inputs)
    x = inception_ResNet_A(x)
    x = reduction_A(x)
    x = inception_resNet_B(x)
    x = reduction_B(x)
    x = inception_resNet_C(x)
    # Classification block
    x = GlobalAveragePooling2D(name='AvgPool')(x)
    x = Dropout(1.0 - 0.8, name='Dropout')(x)
    # Bottleneck
    x = Dense(128, use_bias=False, name='Bottleneck')(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001,
                           scale=False, name='Bottleneck_BatchNorm')(x)

    # Create model
    model = Model(inputs, x, name='inception_resnet_v1')

    return model


if __name__ == "__main__":

    model = InceptionResNetV2()
    model.summary()
