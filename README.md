
## Code for Masters Thesis

Main Reference:
> MuRP: Multi-relational link prediction in the Poincaré ball model of hyperbolic space.
> [[Paper]](https://arxiv.org/pdf/1905.09791.pdf)
> [[Code]](https://github.com/ibalazevic/multirelational-poincare)
>
> TransE: Translating Embeddings for Modeling Multi-relational Data
> [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
> [[Code]](https://github.com/Cesar456/transEwithTorch)
>
> Distmult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases
> [[Paper]](https://arxiv.org/pdf/1412.6575.pdf)
>
> Node-Embedding Project
> [[Code]](https://github.com/kkteru/node-embeddings)


### Running a model

To run the model, execute the following command:

     python main.py 

Description of parameters:
     
     --model poincare
> Choose: poincare, transE or distmult for training
     
     --lr 50
> For poincare model, learning rate is 50, and for transE or distmult model, learning rate is 0.01
     
     --p_norm 1
> This is for transE model, choose 1 or 2

Available datasets are:
    
    FB15k-237
    WN18RR
    NELL-995-h100

### Analysis
Data_Analysis_Khs.py:

> Analyze khs value of a specified relation in the dataset

Data_Analysis_EntityDegree.py:
    
    layered_semantics(dataset = 'NELL-995-h100')
> Get embedding for entities of High/Mid/Low degree 
    
    diff_ways(dim = 20)
> Get entity embedding trained by different model
    
    specify_label()
> Get entity embedding of specified labels

Embedding_PCA.py:

> input entity embedding and get PCA projection.

### Matlab:

Coordinates.m
> Get the scatter plot for PCA projection. 