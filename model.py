#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch
import numpy as np

from parameters import *

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))
    

    def __init__(self, embed_size, hidden_size, num_layers, dropout, attention_size, sourceWord2ind, targetWord2ind, startToken, endToken, unkToken, padToken):
        super(NMTmodel, self).__init__()
        ### Word dictionaries
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_size = attention_size
        self.sourceWord2ind = sourceWord2ind
        self.targetWord2ind = targetWord2ind
        self.targetWords = list(self.targetWord2ind.keys())
        self.targetWordsLen = len(self.targetWords)
        ### Encoder ###
        self.embed_E = torch.nn.Embedding(len(self.sourceWord2ind), self.embed_size)
        self.lstm_E = torch.nn.LSTM(self.embed_size, self.hidden_size, self.num_layers//2, bidirectional=True)
        self.dropout_output_E = torch.nn.Dropout(self.dropout)
        ### Decoder ###
        self.embed_D = torch.nn.Embedding(len(self.targetWord2ind), self.embed_size)
        self.lstm_D = torch.nn.LSTM(self.embed_size, self.hidden_size, self.num_layers)
        self.projection_D = torch.nn.Linear(3*self.hidden_size, len(self.targetWord2ind))
        self.dropout_output_D = torch.nn.Dropout(self.dropout)
        ### Attention ###
        self.W = torch.nn.Linear(3*self.hidden_size, self.attention_size)
        self.v = torch.nn.Linear(self.attention_size, 1)
        ### Tokens ###
        self.startTokenIdx = self.sourceWord2ind[startToken]
        self.endTokenIdx = self.sourceWord2ind[endToken]
        self.unkTokenIdx = self.sourceWord2ind[unkToken]
        self.padTokenIdx = self.sourceWord2ind[padToken]

    def attention(self, H_E, h_Di):
        h = torch.cat((H_E, h_Di.repeat(H_E.shape[0], 1, 1)), dim=2) # (seq_len, batch, 3*self.hidden_size)
        e = self.v(torch.tanh(self.W(h))) #  (seq_len, batch, 1)
        alpha = torch.nn.functional.softmax(torch.transpose(e, 0, 1), dim=1) # (batch, seq_len, 1)
        a = torch.bmm(torch.transpose(alpha, 1, 2), torch.transpose(H_E, 0, 1)) # (batch, 1, 2*self.hidden_size)
        return torch.transpose(a, 0, 1) # (1, batch, 2*self.hidden_size)   

    def forward(self, source, target):
        ### Encoder
        X_E = self.preparePaddedBatch(source, self.sourceWord2ind)
        X_E_embed = self.embed_E(X_E)
        output_E, (h_En, c_En) = self.lstm_E(X_E_embed)
        output_E = self.dropout_output_E(output_E)
        ### Decoder
        X_D = self.preparePaddedBatch(target, self.targetWord2ind)
        X_D_embed = self.embed_D(X_D)
        seq_len = X_D.shape[0]
        batch = X_D.shape[1]

        h_D0 = h_En
        c_D0 = c_En
        output_D, _ = self.lstm_D(X_D_embed, (h_D0, c_D0))
        output_D = self.dropout_output_D(output_D)
        proj_D = torch.zeros(seq_len, batch, 3*self.hidden_size, device=device)
        for i in range(seq_len):
            a = self.attention(output_E, output_D[i].unsqueeze(0))
            proj_D[i] = torch.cat((output_D[i].unsqueeze(0), a), dim=2)
        proj_D = self.projection_D(proj_D)
        
        H = torch.nn.functional.cross_entropy(proj_D[:-1].flatten(0,1), X_D[1:].flatten(0,1), ignore_index=self.padTokenIdx)
        return H

    # def translateSentence(self, sentence, limit=1000):
    #     self.eval()   
    #     with torch.no_grad():
    #         X_E = self.preparePaddedBatch([sentence], self.sourceWord2ind)
    #         X_E_embed = self.embed_E(X_E)
    #         output_E, (h_En, c_En) = self.lstm_E(X_E_embed)
    #         output_E = self.dropout_output_E(output_E)

    #         h_Di = h_En
    #         c_Di = c_En
    #         result = []
    #         wordIdx = self.startTokenIdx
    #         for i in range(limit):
    #             X_D_embed = self.embed_D(torch.tensor([[wordIdx]], device=device))
    #             output_D, (h_Di, c_Di) = self.lstm_D(X_D_embed, (h_Di, c_Di))
    #             output_D = self.dropout_output_D(output_D)
    #             a = self.attention(output_E, output_D)
    #             proj_D = self.projection_D(torch.cat((output_D, a), dim=2))
    #             wordIdx = torch.argmax(torch.nn.functional.softmax(proj_D, dim=2).flatten(0))
    #             if wordIdx == self.endTokenIdx:
    #                 break
    #             result.append(self.targetWords[wordIdx])
    #     return result

    class Node:
        def __init__(self, parent, logprob, wordIdx, depth):
            self.parent = parent
            self.logprob = logprob
            self.hidden_state = None
            self.wordIdx = wordIdx
            self.depth = depth

        def setHiddenState(self, hidden_state):
            self.hidden_state = hidden_state

        def isTerminal(self, endTokenIdx):
            return self.wordIdx == endTokenIdx
    
    def translateSentence(self, sentence, limit=1000):
        self.eval()
        with torch.no_grad():
            X_E = self.preparePaddedBatch([sentence], self.sourceWord2ind)
            X_E_embed = self.embed_E(X_E)
            output_E, state_E = self.lstm_E(X_E_embed)
            output_E = self.dropout_output_E(output_E)

            root = NMTmodel.Node(None, 0, self.startTokenIdx, 0)
            bestNodes = [root]
            maxDepth = root.depth
            terminalNodes = 0
            while terminalNodes < beta and maxDepth < limit:
                currentNodes = []
                for node in bestNodes:
                    if node.isTerminal(self.endTokenIdx):
                        currentNodes.append(node)
                        continue
                    word = node.wordIdx
                    X_D_embed = self.embed_D(torch.tensor([[word]], device=device))
                    state_D = state_E if node.parent == None else node.parent.hidden_state
                    output_D, state_D = self.lstm_D(X_D_embed, state_D)
                    output_D = self.dropout_output_D(output_D)
                    node.setHiddenState(state_D)
                    a = self.attention(output_E, output_D)
                    proj_D = self.projection_D(torch.cat((output_D, a), dim=2))
                    probs, wordIndices = torch.topk(torch.nn.functional.log_softmax(proj_D, dim=2).flatten(0), beta)
                    wordIndices = wordIndices.tolist()
                    probs = probs.tolist()
                    for i in range(len(wordIndices)):
                        currentNodes.append(NMTmodel.Node(node, node.logprob + probs[i], wordIndices[i], node.depth + 1))
                bestNodes = sorted(currentNodes, key=lambda node: node.logprob, reverse=True)[:beta]
                terminalNodes = 0
                for node in bestNodes:
                    terminalNodes += node.isTerminal(self.endTokenIdx)
                if terminalNodes < beta:
                    maxDepth += 1
            bestNode = bestNodes[0]

            result = []
            if maxDepth != limit:
                bestNode = bestNode.parent
            while bestNode.parent != None:
                result.append(self.targetWords[bestNode.wordIdx])
                bestNode = bestNode.parent
            result.reverse()
            return result