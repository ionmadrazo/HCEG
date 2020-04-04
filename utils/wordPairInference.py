import os
import logging
import torch

def normalize(embMatrix):
    norm = embMatrix.norm(p=2, dim=1, keepdim=True)
    result = embMatrix.div(norm.expand_as(embMatrix))
    return result

def NN(sourceEmbs,targetEmbs,inferenceChunkSize=100,IR_retrievalWindow=None):
    """
    Uses traditional nearest neighbour retrieval with cosine similarity
    :param dirs:
    :return: a list containing paired indexes
    """
    pairedIdx=[]
    sourceEmbsNormalized= normalize(sourceEmbs)
    targetEmbsNormalized= normalize(targetEmbs)
    for i,sourceEmbschunk in (enumerate(torch.split(sourceEmbsNormalized,inferenceChunkSize,dim=0))):
        offset= i * inferenceChunkSize
        cosineSims = torch.matmul(sourceEmbschunk,targetEmbsNormalized.t())

        if IR_retrievalWindow is None:
            predictedIdx = torch.argmax(cosineSims,1)
            for idx in range(predictedIdx.size(0)):
                idx1=idx+offset
                idx2=predictedIdx[idx]
                pairedIdx.append((idx1,idx2))

        else:
            for idx in range(cosineSims.size(0)):
                idx1=idx+offset
                start_idx= max(0,idx1-IR_retrievalWindow)
                end_idx=min(cosineSims.size(1),idx1+IR_retrievalWindow)
                idx2=torch.argmax(cosineSims[idx][start_idx:end_idx])
                pairedIdx.append((idx1,idx2))
    return pairedIdx

def computeNeighborhoodDensities(sourceEmbs,targetEmbs,inferenceChunkSize=100,k=10,useFreqBasedHeuristic=False):
    allChunks=[]
    for i,sourceEmbschunk in (enumerate(torch.split(sourceEmbs,inferenceChunkSize,dim=0))):
        offset= i * inferenceChunkSize
        cosineSims = torch.matmul(sourceEmbschunk,targetEmbs.t())
        topkValues,_ = torch.topk(cosineSims,k,dim=1)
        densities = torch.mean(topkValues,1)
        allChunks.append(densities)
    return torch.cat(allChunks,0)

def CSLS(sourceEmbs,targetEmbs,inferenceChunkSize=100,k=10,IR_retrievalWindow=None):
    """
    Uses cross-domain similarity with local scaling for retreival
    :param dirs:
    :return: a list containing paired indexes
    """

    pairedIdx=[]
    sourceEmbsNormalized= normalize(sourceEmbs)
    targetEmbsNormalized= normalize(targetEmbs)

    #sourceNeighbourhoodDensities=computeNeighborhoodDensities(sourceEmbs,targetEmbs,inferenceChunkSize,k=k)
    targetNeighbourhoodDensities=computeNeighborhoodDensities(targetEmbsNormalized,sourceEmbsNormalized,inferenceChunkSize,k=k)

    for i,sourceEmbschunk in (enumerate(torch.split(sourceEmbsNormalized,inferenceChunkSize,dim=0))):
        offset= i * inferenceChunkSize
        cosineSims = torch.matmul(sourceEmbschunk,targetEmbsNormalized.t())
        #Correction only affected by targetNeighbourhoodDensities as sourceNeighbourhoodDensities would be a constant for computing the maximum
        cosineSims = cosineSims-targetNeighbourhoodDensities
        #predictedIdx = torch.argmax(cosineSims,1)
        #for idx in range(predictedIdx.size(0)):
        #    idx1=idx+offset
        #    idx2=predictedIdx[idx]
        #    pairedIdx.append((idx1,idx2))

        if IR_retrievalWindow is None:
            predictedIdx = torch.argmax(cosineSims,1)
            for idx in range(predictedIdx.size(0)):
                idx1=idx+offset
                idx2=predictedIdx[idx]
                pairedIdx.append((idx1,idx2))

        else:
            for idx in range(cosineSims.size(0)):
                idx1=idx+offset
                start_idx= max(0,idx1-IR_retrievalWindow)
                end_idx=min(cosineSims.size(1),idx1+IR_retrievalWindow)
                idx2=torch.argmax(cosineSims[idx][start_idx:end_idx])
                pairedIdx.append((idx1,idx2))
    return pairedIdx
