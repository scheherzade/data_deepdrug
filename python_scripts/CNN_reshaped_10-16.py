#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:45:08 2018

@author: sshirzad
"""
import glob
import cv2
import pickle
import numpy as np
import tensorflow as tf
from rdkit.Chem import PandasTools
import random

##########################################################################
#parameters
##########################################################################   
# Python optimisation variables
learning_rate = 0.0001
epochs = 20
batch_size = 256
#now = datetime.datetime.now()
#date_str=now.strftime("%Y-%m-%d-%H%M")
date_str='2018-10-16-0918'
start_set=0
end_set=1
repeat=1
##########################################################################  
##################################
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
##################################
        
##########################################################################
#generate images
##########################################################################
generate_images=False
generate_ligand=False
directory='/home/sshirzad/workspace/deepdrug/'
savepath=directory+'train_test_data/'+date_str
dpi=50
size=128

################################################################################################
#receptor
################################################################################################ 

if generate_ligand:
    from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
    from gensim.models import word2vec

    model = word2vec.Word2Vec.load('/home/sshirzad/src/mol2vec/examples/models/model_300dim.pkl')

    ligands_folder = glob.glob(directory + 'screen-libs-sdf/*.sdf')

    d_mols={}
    l_num=1
    r_num=1
    for fname in ligands_folder:   
        if 'actives' in fname:
            receptor_name=fname.split('-actives')[0].split('/')[-1]   
            label=1           
        elif 'decoys' in fname:
            receptor_name=fname.split('-decoys')[0].split('/')[-1]
            label=0            
        if receptor_name+'_'+str(label) not in d_mols.keys():
            d_mols[receptor_name+'_'+str(label)]=[]
            
        df = PandasTools.LoadSDF(fname)
        df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
        df['mol2vec'] = [DfVec(x) for x in sentences2vec(df['sentence'], model, unseen='UNK')]
        X = np.array([x.vec for x in df['mol2vec']])
        d_mols[receptor_name+'_'+str(label)]=X

        print(str(l_num), " th receptor")
        l_num = l_num+1

    save_obj(d_mols, directory + 'train_test_data/'+date_str+'/ligand_dict_mols')
else:
    ligand_dict=load_obj(savepath+'/ligand_dict_mols')

#####################################################
#Data
#####################################################             
if generate_images:
    receptors = sorted(glob.glob(directory + 'pockets_dude_tiff_128/'+date_str+'/*.png'))
    d={}
    i=1
    for filename in receptors:
        receptor=filename.split('/')[-1].split('_')[0].split('.')[0]
        
        if receptor!='hxk4':
            print(str(i)+"- voronoi image for "+receptor+" generated")
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            d[receptor]=np.reshape(img,[-1, size*size*3])
        i=i+1
    save_obj(d, savepath+'/receptors_dict_reshaped_'+str(size))
else:
    image_dict=load_obj(savepath+'/receptors_dict_reshaped_'+str(size))
#####################################################
#ligand
#####################################################
receptor_list=sorted(list(set([k.split('_')[0] for k in ligand_dict.keys()])))

all_labels=[]
for k in ligand_dict.keys():
    if k.split('_')[0] in image_dict.keys():
        if k.split('_')[1]=='1':
            all_labels=all_labels+np.shape(ligand_dict[k][0])[0]*[1]
        if k.split('_')[1]=='0':
            all_labels=all_labels+np.shape(ligand_dict[k][0])[0]*[0]


from sklearn.utils import class_weight
weights=class_weight.compute_class_weight('balanced', [0,1],all_labels)

###########################################################################################
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]
    
    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    
    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    
    # add the bias
    out_layer += bias
    
    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)
    
    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')
    
    return out_layer


# declare the training data placeholders
# input x - for 20 x 20 pixels = 400 - this is the flattened image data t
x = tf.placeholder(tf.float32, [None,size*size*3])
# dynamically reshape the input
x_shaped = tf.reshape(x, [-1, size, size, 3])
# now declare the output data placeholder - 2 classes
y = tf.placeholder(tf.float32, [None, 2])

# create some convolutional layers
layer1 = create_new_conv_layer(x_shaped, 3, 16, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 16, 32, [3, 3], [2, 2], name='layer2')
layer3 = create_new_conv_layer(layer2, 32, 64, [3, 3], [2, 2], name='layer3')

flattened = tf.reshape(layer3, [-1, size * size])  #(size/8)*(size/8)*64
# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([size * size , 512], stddev=0.03), name='wd1') #(size/8)*(size/8)*64
bd1 = tf.Variable(tf.truncated_normal([512], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

z = tf.placeholder(tf.float32, [None,300])
z_shaped = tf.reshape(z, [-1, 300])
x_final=tf.concat([dense_layer1,z_shaped],-1)


wd2 = tf.Variable(tf.truncated_normal([512+300, 1000], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(x_final, wd2) + bd2
dense_layer2 = tf.nn.relu(dense_layer2)

# another layer with softmax activations
wd3 = tf.Variable(tf.truncated_normal([1000, 2], stddev=0.03), name='wd3')
bd3 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name='bd3')
dense_layer3 = tf.matmul(dense_layer2, wd3) + bd3
y_ = tf.nn.softmax(dense_layer3)

class_weights=tf.constant([[  0.51474333,  17.45682248]])
weights = tf.reduce_sum(class_weights * y, axis=1)
# compute your (unweighted) softmax cross entropy loss
unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer3, labels=y)
# apply the weights, relying on broadcasting of the multiplication
weighted_losses = unweighted_losses * weights
# reduce the result to get your final loss
loss = tf.reduce_mean(weighted_losses)
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer3, labels=y))

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred = tf.cast(tf.argmax(y_, 1),tf.float32)
label = tf.cast(tf.argmax(y,1),tf.float32)
auc=tf.metrics.auc(pred, label,curve='ROC')
confusion = tf.confusion_matrix(labels=label, predictions=pred, num_classes=2)
###########################################################################################


for set_i in range(start_set, end_set):
    test_receptor=receptor_list[set_i]  
    print('test receptor: ',test_receptor)
    
    training_set_0=[]
    training_set_1=[]
    test_set_0=[]
    test_set_1=[]
    
    for k in ligand_dict.keys():
        if k.split('_')[0] in image_dict.keys():
            if k.split('_')[0]!=test_receptor:
                if k.endswith('_0'):
                    for r in ligand_dict[k][0]:
                        r_reshaped=np.reshape(np.asarray(r),[-1,300])
                        training_set_0.append(np.hstack((r_reshaped,image_dict[k.split('_')[0]])))
                else:
                    for r in ligand_dict[k][0]:
                        r_reshaped=np.reshape(np.asarray(r),[-1,300])
                        training_set_1.append(np.hstack((r_reshaped,image_dict[k.split('_')[0]])))
            else:
                if k.endswith('_0'):
                    for r in ligand_dict[k][0]:
                        r_reshaped=np.reshape(np.asarray(r),[-1,300])                    
                        test_set_0.append(np.hstack((r_reshaped,image_dict[k.split('_')[0]])))
                else:
                    for r in ligand_dict[k][0]:
                        r_reshaped=np.reshape(np.asarray(r),[-1,300])
                        test_set_1.append(np.hstack((r_reshaped,image_dict[k.split('_')[0]])))
        else:
            print("error, "+k+" key does not exist in image_dict")
    const=4
    train_size=int(len(training_set_0)/const)+int(len(training_set_1)/const)
    test_size=len(test_set_0)+len(test_set_1)
    training_set=np.zeros((train_size, 300 + size*size*3))
    training_labels=np.zeros((train_size,2))

    temp=random.sample(training_set_1, int(len(training_set_1)/const))

    temp_size=np.shape(temp)[0]
    for p in range(temp_size):
#        training_set[p,:]=training_set_1[p]
        training_set[p,:]=temp[p]
        training_labels[p,1]=1
        
    temp=random.sample(training_set_0, int(len(training_set_0)/const))

    for p in range(len(temp)):
        training_set[temp_size+p,:]=training_set_0[p]
        training_labels[temp_size+p,0]=1
        

    test_set=np.zeros((test_size, 300 + size*size*3))
    test_labels=np.zeros((test_size, 2))

    for p in range(len(test_set_1)):
        test_set[p,:]=test_set_1[p]
        test_labels[p,0]=1        
        
    for p in range(len(test_set_1)):
        test_set[len(test_set_1)+p,:]=training_set_1[p]
        test_labels[len(test_set_1)+p,1]=1
    
    per = np.random.permutation(train_size)

    training_set=training_set[per]    
    training_labels=training_labels[per]
    
    per = np.random.permutation(test_size)

    test_set=test_set[per]    
    test_labels=test_labels[per]
           
    # setup the initialisation operator
    init_op = tf.global_variables_initializer()
    init_loc_op = tf.local_variables_initializer()
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        sess.run(init_loc_op)
        total_batch = int(len(training_labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x = training_set[i*batch_size:(i+1)*batch_size,300:]
                batch_y =training_labels[i*batch_size:(i+1)*batch_size] 
#                print(np.sum(test_labels[:,0]),'   ',np.sum(test_labels[:,1]))
                batch_z =training_set[i*batch_size:(i+1)*batch_size,0:300]
    
                _, c = sess.run([optimiser, loss], feed_dict={x: batch_x, y: batch_y, z: batch_z})
    
                avg_cost += c / total_batch
            test_acc = sess.run(accuracy, feed_dict={x: test_set[:,300:], y: test_labels,  z: test_set[:,0:300]})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
            print("Epoch:", (epoch + 1), "confusion matrix: ",sess.run(confusion, feed_dict={x: test_set[:,300:], y: test_labels,  z: test_set[:,0:300]}))

        
        print("\nTraining complete!")
#        print(sess.run(accuracy, feed_dict={x: test_set[:,300:], y: test_labels,  z: test_set[:,0:300]}))
        print(sess.run(auc, feed_dict={x: test_set[:,300:], y: test_labels,  z: test_set[:,0:300]}))
#        print(sess.run(confusion, feed_dict={x: test_set[:,300:], y: test_labels,  z: test_set[:,0:300]}))
