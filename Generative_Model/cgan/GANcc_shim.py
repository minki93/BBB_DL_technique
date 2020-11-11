def GANcc(Xr,oneC,yr,Yclass, n_samplings, z_dim, learning_rate, batch_size, epochs, d_hidden_size=128, g_hidden_size=128):
    import numpy as np
    import tensorflow as tf
    tf.reset_default_graph()
    sample_size=Xr.shape[0] 
    x_dim = Xr.shape[1]
    
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    keepprob =1
       
    y_dim=len(np.intersect1d(yr,yr))
    Yr=np.zeros((sample_size,y_dim))
    for ii in range(sample_size):
        Yr[ii, yr[ii]]=oneC
    
    
# Discriminator Net
    X = tf.placeholder(tf.float32, shape=[None, x_dim], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, y_dim], name='Y')
    D_W1=tf.get_variable('D_W1',shape=[x_dim+y_dim,d_hidden_size],initializer=tf.contrib.layers.xavier_initializer())
    D_b1 = tf.Variable(tf.random_normal([d_hidden_size]), name='D_b1')
    D_W2 = tf.get_variable('D_W2',shape=[d_hidden_size,1],initializer=tf.contrib.layers.xavier_initializer())
    D_b2 = tf.Variable(tf.random_normal([1]), name='D_b2')
    theta_D = [D_W1, D_W2, D_b1, D_b2]


# Generator Net
    Z = tf.placeholder(tf.float32, shape=[None, z_dim], name='Z')
    G_W1 = tf.get_variable('G_W1',shape=[z_dim+y_dim,g_hidden_size],initializer=tf.contrib.layers.xavier_initializer())
    G_b1 = tf.Variable(tf.zeros(shape=[g_hidden_size]), name='G_b1')
    G_W2 = tf.get_variable('G_W2',shape=[g_hidden_size,x_dim],initializer=tf.contrib.layers.xavier_initializer())
    G_b2 = tf.Variable(tf.zeros(shape=[x_dim]), name='G_b2')
    theta_G = [G_W1, G_W2, G_b1, G_b2]

    def discriminator(x,y):
    # Leaky ReLU
        xy=tf.concat([x,y],axis=1)
        h1=tf.matmul(xy, D_W1) + D_b1        
        h1 = tf.maximum(h1,0)    
        h1=tf.nn.dropout(h1, keep_prob)
        h2 = tf.matmul(h1, D_W2) + D_b2
        h2=tf.nn.dropout(h2, keep_prob)
        prob = tf.nn.sigmoid(h2)
        return prob, h2


    def generator(z,y):
    # Leaky ReLU
        zy=tf.concat([z,y],axis=1)
        h1=tf.matmul(zy, G_W1) + G_b1        
        h1 = tf.maximum(h1,0) 
        h2 = tf.matmul(h1, G_W2) + G_b2
        out = tf.nn.tanh(h2) 
        return out


    G_z = generator(Z,Y)
    D_real, D_logit_real = discriminator(X,Y)
    D_fake, D_logit_fake = discriminator(G_z,Y)
#D_loss_real = -tf.reduce_mean(tf.log(D_real+1e-8)); D_loss_fake=-tf.reduce_mean( tf.log(1. - D_fake+1e-8))
#D_loss=D_loss_real+D_loss_fake;  G_loss = -tf.reduce_mean(tf.log(D_fake+1e-8))
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))


    D_solver = tf.train.AdamOptimizer(learning_rate,beta1=0.1).minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(learning_rate,beta1=0.1).minimize(G_loss, var_list=theta_G)

    sess=tf.Session(); sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        total_batch=int(np.ceil(sample_size/batch_size))
        avg_loss=0
        for ii in range(total_batch):
            if ii < int(sample_size/batch_size):
                  batch_X=Xr[ii*batch_size:(ii+1)*batch_size]    
                  batch_Y=Yr[ii*batch_size:(ii+1)*batch_size] 
            else:
                  batch_X=Xr[ii*batch_size:]         
                  batch_Y=Yr[ii*batch_size:]
        
            colsum=np.sum(batch_Y,axis=0)
            if np.min(colsum)==0: continue
            
            batch_z = np.random.uniform(-1, 1, size=(batch_X.shape[0], z_dim)) 
            D_loss_curr,_ = sess.run([D_loss,D_solver], feed_dict={X:batch_X, Z: batch_z, Y: batch_Y,keep_prob:keepprob})
            G_loss_curr,_ = sess.run([G_loss,G_solver], feed_dict={X:batch_X, Z: batch_z, Y:batch_Y,keep_prob:keepprob})    
            avg_loss+=(D_loss_curr+G_loss_curr)/total_batch
            
    sample_z = np.random.uniform(-1, 1, size=(n_samplings,z_dim))
    sample_y = np.zeros(shape=[n_samplings, y_dim]);        

    sample_y[:,Yclass]=oneC
            
    G_z = sess.run(G_z,feed_dict={X:Xr, Z:sample_z,Y:sample_y})
    
    return G_z #, avg_loss

def GAN_SMILES(data, oneC, yr, Yclass, n_samplings, learning_rate,batch_size,epochs,xchange):
    import numpy as np
    import tensorflow as tf
    tf.reset_default_graph()
# for eg, BR to R, Cl to L
    if len(xchange)>0:
       for i in range(len(xchange)):
           data=data.replace(xchange[i][0],xchange[i][1]);  
    chars = list(set(data)); 
    if chars.count('\n')>0: chars.remove('\n');
    if chars.count(' ')>0: chars.remove(' ');
    char_to_int = { ch:i+1 for i,ch in enumerate(chars) }
    int_to_char = { i+1:ch for i,ch in enumerate(chars) }
    xc=data.split()
    str_len=max(len(x) for x in xc)     
    n_samples, n_chars = len(xc), len(chars) 
    n_chars1=n_chars+1 # space 포함    

    x_dim = str_len*n_chars1
    z_dim = x_dim

    x=np.zeros((n_samples,str_len), dtype=np.int32)+n_chars1
    for ii in range(len(xc)):
        for i in range(len(xc[ii])):
            x[ii,i]=char_to_int[xc[ii][i]]
          
# Xr = n_samples*(str_len*n_chars1)
    Xr=np.zeros((n_samples, str_len*n_chars1), dtype=np.int32)
    for ii in range(Xr.shape[0]):
        v1=np.zeros((1,str_len*n_chars1),dtype=np.int32 )
        for i in range(str_len):
            v=np.zeros((1,n_chars1),dtype=np.int32 )
            v[0,x[ii][i]-1]=1
            if i==0:
                v1=v
            else:
                v1=np.concatenate((v1,v),axis=1)
        Xr[ii,:]=v1
    

    gen_samples0= GANcc(Xr,oneC,yr,Yclass, n_samplings, z_dim, learning_rate, batch_size, epochs)
      
             
    
    gen_samples=['']*n_samplings

    for jj in range(n_samplings):
        a1=gen_samples0[jj].reshape((str_len, n_chars1))
        a1=a1[0:str_len,:]
        a2=np.zeros((str_len, n_chars1), dtype=np.int32)
        for ii in range(str_len):
              a2[ii,np.argmax(a1[ii])]=np.argmax(a1[ii])+1         
        a3=''
        for ii in range(str_len):
            if max(a2[ii,:]) < n_chars1:
                a3+=int_to_char[max(a2[ii,:])]
            else:     
                a3+=''
        gen_samples[jj]=a3

    
    # for eg, R to Br, L to Cl
    for ii in range(len(gen_samples)): 
        if len(xchange)>0:
            for i in range(len(xchange)):
                gen_samples[ii]=gen_samples[ii].replace(xchange[i][1],xchange[i][0])
    
    from rdkit import Chem
    vc=list()
    for ii in range(len(gen_samples)):
        m=Chem.MolFromSmiles(gen_samples[ii])
        if m is None: continue
        else: vc.append(ii); 
# vc = 유효한 SMILES structure인  gen_samples의 번호             

    gen_samplesF=['']*len(vc)
    if len(vc)==0:
        gen_samplesF=''    
    else:
        for ii in range(len(vc)):
            gen_samplesF[ii]=gen_samples[vc[ii]]
            
    
    gen_samplesF=set(gen_samplesF) & set(gen_samplesF) # 중복되는 structure 제거 
    gen_samplesF= set(gen_samplesF) - set(xc) # tr data와 같은 structure 제거 
    gen_samplesF=list(gen_samplesF) # list type로 
      
    if len(gen_samplesF)==0:
        print('No valid SMILES structure is generated.')   
    else:
        for ii in range(len(gen_samplesF)):
            print('Vailid SMILES structure generated',ii+1,': ',gen_samplesF[ii],'of length',len(gen_samplesF[ii]))  
    return gen_samplesF 