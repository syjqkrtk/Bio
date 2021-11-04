import random
import numpy as np
# This file contains the dataset in a useful way. We populate a list of
# Trees to train/test our Neural Nets such that each Tree contains any
# number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..

embedding = {}

class Node:  # a node in the tree
    def __init__(self, label, word=None, data=[0,0,0,0,0,0,0,0,0]):
        self.label = label
        self.word = word
        self.data = data
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)

    def __str__(self):
        if self.isLeaf:
            return '[{0}:{1}]'.format(self.word, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)


class Tree:

    def __init__(self, treeString, start, openChar='(', closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        self.start = start
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        # get list of labels as obtained through a post-order traversal
        self.labels = get_labels(self.root)
        self.num_words = len(self.labels)
              

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2  # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1]))  # zero index labels

        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = str(int(''.join(tokens[2:-1]).lower())+self.start)  # lower case?
            #node.word = str(int(''.join(tokens[2:-1]).lower())%1000)  # lower case?
            node.data = embedding[node.word]  # lower case?
            #print(node.data)
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node)

        return node

    def get_name(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return str(int(float(words[0])/1000000))+","+str(int(float(words[0])/10000)%100)

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words

    def get_data(self):
        leaves = getLeaves(self.root)
        data = [node.data for node in leaves]
        return data


def leftTraverse(node, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    if node is None:
        return
    leftTraverse(node.left, nodeFn, args)
    leftTraverse(node.right, nodeFn, args)
    nodeFn(node, args)


def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)


def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]


def clearFprop(node, words):
    node.fprop = False

def loadTrees(dataSet='train', start=0):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    file = 'trees/%s.txt' % dataSet
    print("Loading %s trees.." % dataSet)
    with open(file, 'r') as fid:
        trees = [Tree(l,start=start) for l in fid.readlines()]

    return trees

def dict_embedding(file, start):
    file2 = open(file,'r')
    temp = file2.readline().replace("\n","")
    end = 0 

    while temp:
        temp2 = temp.split(",")
        embedding[str(int(temp2[0])+start)] = [np.abs(float(t)) for t in temp2[1:-1]]
        #embedding[str(int(temp2[0])%1000)] = [np.abs(float(t)) for t in temp2[1:-1]]
        temp = file2.readline().replace("\n","")
        end = max(end,int(temp2[0]))
    file2.close()

    return 10000*(np.ceil(end/10000))

def simplified_data(num_train, num_dev, num_test, p, q):
    rndstate = random.getstate()
    random.seed(0)
    #trees = loadTrees('train') + loadTrees('dev') + loadTrees('test')
    print("Loading trees/Text/embed_word.csv embeddings..")
    end = dict_embedding('trees/Text/SNP/embed_word_'+str(p)+'.csv',0)
    dict_embedding('trees/Text/SNP/embed_word_'+str(q)+'.csv',end)
    trees1 = loadTrees('Text/SNP/RE_'+str(p),0)
    trees2 = loadTrees('Text/SNP/RE_'+str(q),end)
    
    #filter extreme trees
    pos_trees1 = [t for t in trees1 if (t.root.label==3) or (t.root.label==2)]
    neg_trees1 = [t for t in trees1 if (t.root.label==1) or (t.root.label==0)]
    pos_trees2 = [t for t in trees2 if (t.root.label==3) or (t.root.label==2)]
    neg_trees2 = [t for t in trees2 if (t.root.label==1) or (t.root.label==0)]

    #binarize labels
    binarize_labels(pos_trees1)
    binarize_labels(neg_trees1)
    binarize_labels(pos_trees2)
    binarize_labels(neg_trees2)
    
    #split into train, dev, test
    print(len(pos_trees1), len(neg_trees1))
    pos_trees1 = sorted(pos_trees1, key=lambda t: len(t.get_words()))
    neg_trees1 = sorted(neg_trees1, key=lambda t: len(t.get_words()))
    pos_trees2 = sorted(pos_trees2, key=lambda t: len(t.get_words()))
    neg_trees2 = sorted(neg_trees2, key=lambda t: len(t.get_words()))
    num_train=int(num_train/2)
    num_dev=int(num_dev/2)
    num_test=int(num_test/2)
    #random.shuffle(pos_trees)
    #random.shuffle(neg_trees)
    train = pos_trees1[:num_train] + neg_trees1[:num_train]
    dev = pos_trees2[:num_dev] + neg_trees2[:num_dev]
    test = pos_trees1 + neg_trees1 + pos_trees2 + neg_trees2
    #train = pos_trees[:num_train] + neg_trees[:num_train]
    #dev = pos_trees[num_train : num_train+num_dev] + neg_trees[num_train : num_train+num_dev]
    #test = pos_trees[num_train+num_dev : num_train+num_dev+num_test] + neg_trees[num_train+num_dev : num_train+num_dev+num_test]
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    random.setstate(rndstate)


    return train, dev, test


def binarize_labels(trees):
    def binarize_node(node, _):
        if node.label==0:
            node.label = 0
        elif node.label==3:
            node.label = 1
        else:
            node.label = -1
    for tree in trees:
        leftTraverse(tree.root, binarize_node, None)
        tree.labels = get_labels(tree.root)
