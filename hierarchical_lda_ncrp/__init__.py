import numpy as np
from scipy.special import gammaln
import random
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import pydot
import itertools
import string

def CRP(topic, gamma):

    crp_p = np.zeros(len(topic)+1)
    M = 0
    for i in range(len(topic)):
        M += len(topic[i])
        
    for i in range(len(topic)+1):
        if i == 0:
            crp_p[i] = gamma / (gamma + M) 
        else:
            crp_p[i] = len(topic[i-1])/(gamma + M)
    return crp_p

def topic_sampling(corpus, gamma):
    topic = []
    flat_words = list(itertools.chain.from_iterable(corpus))
    i = 0
    while i < len(flat_words):
        cm_prop = CRP(topic, gamma)/sum(CRP(topic, gamma))
        theta = np.random.multinomial(1,cm_prop).argmax()
        topic.append([flat_words[i]]) if theta == 0 else topic[theta-1].append(flat_words[i])
        i+=1
    return topic

def Z(corpus, T, alpha, beta):
   
    D = len(corpus)

    num_vocab = 0
    for i in range(D):
        num_vocab += len(corpus[i])
    z_topic=[[] for t in range(T)]
    z_doc=[[] for t in range(T)]
    dict = [[key,i] for i,c in enumerate(corpus) for j, key in enumerate(c)]  
    
    for e in dict:
        wi,i,j,p = e[0],e[1],0,np.zeros(T) 
        while j < T:
            lik=(z_topic[j].count(wi)+beta)/(len(z_topic[j]) +num_vocab*beta)
            pri=(np.sum(np.isin(z_topic[j],corpus[i]))+alpha)/(len(corpus[i]) +T*alpha)
            p[j]=lik * pri 
            j += 1
        i_top = np.random.multinomial(1, p/np.sum(p)).argmax()
        z_topic[i_top].append(wi)
        z_doc[i_top].append(i)
    
    return list(filter(None, z_topic)), list(filter(None, z_doc))

def CRP_prior(corpus, doc_topic, gamma):

    doc_p = np.zeros((len(corpus), len(doc_topic)))
    for i in range(len(corpus)):
        doc = []
        for j in range(len(doc_topic)):
            doc_num = [num for num in doc_topic[j]]
            doc.append(doc_num)
        doc_p[i,:] = CRP(doc, gamma)[1:]
    return doc_p

def word_likelihood(corpus, topic, eta):
    
    wm = np.zeros((len(corpus), len(topic)))
    
    W = 0
    for i in range(len(corpus)):
        W += len(corpus[i])
    
    for i in range(len(corpus)):
        doc = corpus[i]
        for j in range(len(topic)):
            l = topic[j]
            denom_1 = 1
            num_2 = 1
            
            n_cml_m = len(l) - len([w for w in set(doc) if w in l])
            num_1 = gammaln(n_cml_m + W * eta)
            denom_2 = gammaln(len(l) + W * eta)
            
            for word in doc:
                nw_cml_m = l.count(word) - doc.count(word)
                if nw_cml_m <= 0:
                    nw_cml_m = 0
                
                denom_1 += gammaln(nw_cml_m + eta)
                num_2 += gammaln(l.count(word) + eta)
            
            wm[i,j] = num_1 + num_2 - denom_1 - denom_2
        wm[i, :] = wm[i, :] + abs(min(wm[i, :]) + 0.1)
    wm = wm/wm.sum(axis = 1)[:, np.newaxis]
    return wm

def gibbs_sampling(corpus, T , alpha, beta, gamma, eta, ite):
    
    num_vocab = np.sum([len(x) for x in corpus])
    gibbs = np.zeros((num_vocab, ite))
    
    
    for it in range(ite):
        doc_topic= Z(corpus, T, alpha, beta)[0]
        doc_p = CRP_prior(corpus, doc_topic, gamma)
        lik = word_likelihood(corpus, doc_topic, eta)
        c_m = (lik * doc_p) / (lik * doc_p).sum(axis = 1).reshape(-1,1) #posterior
        
        g=[]
        for i in range(len(corpus)):
            if np.sum(c_m[i,:-1])>1:
                c_m[i,:-1]=c_m[i,:-1]/np.sum(c_m[i,:-1])
                c_m[i,-1]=0
            for word in corpus[i]:
                p = np.random.multinomial(1, c_m[i])
                g.append(p.argmax())
        
        gibbs[:,it]=g
    
    t=[]
    for i in range(num_vocab):
        tt = int(Counter(gibbs[i,:]).most_common(1)[0][0])
        t.append(tt)
        
    n_topic=np.max(t)+1

    wn_topic = [[] for _ in range(n_topic)]
    
    n = 0
    for doc in corpus:
        wn_doc_topic = [[] for _ in range(n_topic)]
        for word in doc:
            k = t[n]
            wn_doc_topic[k].append(word)
            n += 1
        for i in range(n_topic):
            if len(wn_doc_topic[i]) != 0:
                k = wn_doc_topic[i]
                wn_topic[i].append(k)

    wn_topic = [x for x in wn_topic if x != []]
    return wn_topic

def hLDA(corpus, alpha, beta, gamma, eta, ite, level,num=3):

    topic = topic_sampling(corpus, gamma)
    topic = len(topic)
    hLDA_tree = [[] for t in range(level)]
    node_num = [[] for t in range(level+1)]
    node_num[0].append(1)
    
    print("***LEVEL 0***\n")
 
    # Initialize the tree:
    wn_topic = gibbs_sampling(corpus, topic, alpha, beta, gamma, eta, ite)
    node_topic = sum(wn_topic[0],[])
    hLDA_tree[0].append(node_topic)
    print_t = [i[0] for i in Counter(node_topic).most_common(num)]
    print('NODE 1:',print_t)
    tmp_tree = wn_topic[1:]
    node_num[1].append(len(wn_topic[1:]))
    
    # Define helper function to expand the hLDA tree
    def expand_hLDA_tree(tmp_tree, hLDA_tree, node_num, i, it):
        j = 0
        while j < it:
            if len(tmp_tree)==0:
                break
            wn_topic1 = gibbs_sampling(tmp_tree[0], topic, alpha, beta, gamma, eta, ite)
            node_topic1 = [n for w in wn_topic1[0] for n in w]
            hLDA_tree[i].append(node_topic1)
            tmp_tree.remove(tmp_tree[0])
            print_t = [i[0] for i in Counter(node_topic1).most_common(num)]
            print('NODE',j+1,":",print_t)
            if wn_topic1[1:] != []: tmp_tree.extend(wn_topic1[1:]) 
            node_num[i+1].append(len(wn_topic1[1:]))
            j+=1
            
    for i in range(1, level): 
        print(' ')
        print("***LEVEL %d***" % i)
        it = sum(node_num[i])
        expand_hLDA_tree(tmp_tree, hLDA_tree, node_num, i, it)
    
    return hLDA_tree, node_num[:level]

def tree_plot(hLDA_object, num = 3, save = False):
    
    from IPython.display import Image, display
    def viewPydot(pdot):
        plt = Image(pdot.create_png())
        display(plt)

    words,struc = hLDA_object
    graph = pydot.Dot(graph_type='graph')
    end_index = [np.insert(np.cumsum(i),0,0) for i in struc]
    
    for level in range(len(struc)-1):
        leaf_word = words[level + 1]
        leaf_struc = struc[level + 1]
        word = words[level]
        end_leaf_index = end_index[level+1]

        def node_plot(leaf_word, leaf_struc, end_leaf_index, word):
            for i,e in enumerate(word):
                root = '\n'.join([x[0] for x in Counter(e).most_common(num)])
                lf = leaf_word[end_leaf_index[i]:end_leaf_index[i+1]]  
                for l in lf:
                    leaf_w = '\n'.join([x[0] for x in Counter(list(l)).most_common(num)])
                    edge = pydot.Edge(root, leaf_w)
                    graph.add_edge(edge)
    
        for w in word:
            node_plot(leaf_word, leaf_struc, end_leaf_index, word)
    
    if save == True:
        graph.write_png('graph.png')
    
    viewPydot(graph)
