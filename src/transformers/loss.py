import logging
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple

logger = logging.getLogger(__name__)

class CosineEmbeddingMSELoss(nn.Module):
    def __init__(self, dim=1, eps=1e-08):
        super(CosineEmbeddingMSELoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=dim, eps=eps)
        self.loss = nn.MSELoss()

    def forward(self, feature_a, feature_b, labels):
        cos = self.similarity(feature_a, feature_b)
        logits = cos.view(-1)
        logits = (cos + 1.0) / 2.0
        loss = self.loss(logits, labels.view(-1))
        return logits, loss

class CosineEmbeddingBCELoss(nn.Module):
    def __init__(self, dim=1, eps=1e-08):
        super(CosineEmbeddingBCELoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=dim, eps=eps)
        self.loss = nn.BCELoss()

    def forward(self, feature_a, feature_b, labels):
        cos = self.similarity(feature_a, feature_b)
        logits = cos.view(-1)
        logits = (cos + 1.0) / 2.0
        loss = self.loss(logits, labels.view(-1))
        return logits, loss

class CosineEmbeddingBCELossWithNegtive(nn.Module):
    def __init__(self, dim=1, eps=1e-08):
        super(CosineEmbeddingBCELossWithNegtive, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=dim, eps=eps)
        self.loss = nn.BCELoss()

    def forward(self, feature_a, feature_b, labels):
        batch_size = feature_a.shape[0] 
        num_of_neg =  batch_size-1
        feature_a_rot=feature_a.repeat(num_of_neg+1, 1)
        feature_b_rot=feature_b.repeat(1,1)
        for i in range(num_of_neg):
            feature_b_rot = torch.cat((
                feature_b_rot, 
                feature_b[i+1:, : ],
                feature_b[0: i+1, : ],
                ), 0)
        cos = self.similarity(feature_a_rot, feature_b_rot)
        logits = cos.view(-1)
        logits = (cos + 1.0) / 2.0
        labels = labels.view(-1)
        labels = torch.cat((labels, torch.zeros(num_of_neg*batch_size).cuda()), 0)
        loss = self.loss(logits, labels.view(-1))
        loss_positive = self.loss(logits[:batch_size-1], labels.view(-1)[:batch_size-1])
        loss_negative = self.loss(logits[batch_size:], labels.view(-1)[batch_size:])
        logger.info("-------batch_size: {0}".format(batch_size))
        logger.info("-------loss_positive: {0}".format(loss_positive))
        logger.info("-------loss_negative: {0}".format(loss_negative))
        return logits, loss

class CosineEmbeddingBCELossSumWithWeightedNegtive(nn.Module):
    def __init__(self, dim=1, eps=1e-08, neg_weight=1):
        super(CosineEmbeddingBCELossSumWithWeightedNegtive, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=dim, eps=eps)
        self.neg_weight = neg_weight

    def forward(self, feature_a, feature_b, labels):
        batch_size = feature_a.shape[0] 
        num_of_neg =  batch_size-1
        feature_a_rot=feature_a.repeat(num_of_neg+1, 1)
        feature_b_rot=feature_b.repeat(1,1)
        for i in range(num_of_neg):
            feature_b_rot = torch.cat((
                feature_b_rot, 
                feature_b[i+1:, : ],
                feature_b[0: i+1, : ],
                ), 0)
        cos = self.similarity(feature_a_rot, feature_b_rot)
        logits = cos.view(-1)
        logits = (cos + 1.0) / 2.0
        labels = labels.view(-1)
        labels = torch.cat((labels, torch.zeros(num_of_neg*batch_size).cuda()), 0)
        weight = torch.cat((torch.ones(batch_size).cuda(), self.neg_weight*torch.ones(num_of_neg*batch_size).cuda()), 0)
        self.loss = nn.BCELoss(weight=weight,reduction='sum')
        
        #print(logits.shape)
        #print(labels.shape)
        
        loss = self.loss(logits, labels.view(-1))
        return logits, loss

class CosineEmbeddingBCELossWithWeightedNegtive(nn.Module):
    def __init__(self, dim=1, eps=1e-08, neg_weight=1):
        super(CosineEmbeddingBCELossWithWeightedNegtive, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=dim, eps=eps)
        self.neg_weight = neg_weight

    def forward(self, feature_a, feature_b, labels):
        batch_size = feature_a.shape[0] 
        num_of_neg =  batch_size-1
        feature_a_rot=feature_a.repeat(num_of_neg+1, 1)
        feature_b_rot=feature_b.repeat(1,1)
        for i in range(num_of_neg):
            feature_b_rot = torch.cat((
                feature_b_rot, 
                feature_b[i+1:, : ],
                feature_b[0: i+1, : ],
                ), 0)
        cos = self.similarity(feature_a_rot, feature_b_rot)
        logits = cos.view(-1)
        logits = (cos + 1.0) / 2.0
        labels = labels.view(-1)
        labels = torch.cat((labels, torch.zeros(num_of_neg*batch_size).cuda()), 0)
        # weight = torch.cat((torch.ones(batch_size).cuda(), self.neg_weight*torch.ones(num_of_neg*batch_size).cuda()), 0)
        weight = torch.cat((self.neg_weight*torch.ones(batch_size).cuda(), torch.ones(num_of_neg*batch_size).cuda()), 0)
        self.loss = nn.BCELoss(weight=weight)
        self.lossp = nn.BCELoss(weight=weight[:batch_size-1])
        self.lossn = nn.BCELoss(weight=weight[batch_size:])
        
        #print(logits.shape)
        #print(labels.shape)
        loss = self.loss(logits, labels.view(-1))
        
        loss_positive = self.lossp(logits[:batch_size-1], labels.view(-1)[:batch_size-1])
        loss_negative = self.lossn(logits[batch_size:], labels.view(-1)[batch_size:])        
        logger.info("-------batch_size: {0}".format(batch_size))
        logger.info("-------loss_positive: {0}".format(loss_positive))
        logger.info("-------loss_negative: {0}".format(loss_negative))

        return logits, loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2, neg_weight=1):
        super(TripletLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)

    def forward(self, feature_a, feature_b):
        batch_size = feature_a.shape[0] 
        num_of_neg =  batch_size-1
        
        feature_anchor = feature_a.repeat(num_of_neg,1)
        feature_positive = feature_b.repeat(num_of_neg,1)
        feature_negtive = torch.cat((
                feature_b[1:, : ],
                feature_b[0:1, : ],
                ), 0)
        for i in range(num_of_neg):
            if i>0:
                feature_negtive = torch.cat((
                    feature_negtive, 
                    feature_b[i+1:, : ],
                    feature_b[0: i+1, : ],
                    ), 0)
        # print("anchor %s" % feature_anchor)
        # print("positive %s" % feature_positive)
        # print("feature_negtive %s" % feature_negtive)
        loss = self.triplet_loss(feature_anchor, feature_positive, feature_negtive)
        return loss

class TripletHardLoss(nn.Module):
    def __init__(self, margin=1.0, p=2, neg_weight=1):
        super(TripletHardLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)

    def forward(self, feature_a, feature_b, feature_c, feature_d):
        batch_size = feature_a.shape[0] 
        num_of_neg =  batch_size-1
        
        feature_anchor = feature_a.repeat(num_of_neg + 2,1)
        feature_positive = feature_b.repeat(num_of_neg + 2,1)
        feature_negtive = torch.cat((
                feature_b[1:, : ],
                feature_b[0:1, : ],
                ), 0)
        for i in range(num_of_neg):
            if i>0:
                feature_negtive = torch.cat((
                    feature_negtive, 
                    feature_b[i+1:, : ],
                    feature_b[0: i+1, : ],
                    ), 0)
        feature_negtive = torch.cat((
            feature_negtive, 
            feature_c,
            feature_d,
            ), 0)
        
        #print("anchor %s" % feature_anchor)
        #print("positive %s" % feature_positive)
        #print("feature_negtive %s" % feature_negtive)
        loss = self.triplet_loss(feature_anchor, feature_positive, feature_negtive)
        return loss

r'''
    criterion = CircleLoss(m=0.25, gamma=256)
    circle_loss = criterion(inp_sp, inp_sn)
'''
class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, encoder_output: Tensor, decoder_output: Tensor) -> Tensor:
        num_of_neg = encoder_output.shape[0] -1
        def calculator_circle_loss(encoder_output, decoder_output):
            decoder_output_rot = decoder_output.repeat(1,1)
            for i in range(num_of_neg):
                decoder_output_rot = torch.cat((
                    decoder_output_rot, 
                    decoder_output[i+1:, : ],
                    decoder_output[0: i+1, : ],
                    ), 0)
            #decoder_output_rot: [(num_of_neg+1)*batchsize, dim]
            encoder_norm = torch.sqrt(torch.sum(torch.square(encoder_output), 1, True))
            encoder_norm = encoder_norm.repeat(num_of_neg+1, 1)
            #encoder_norm: [(num_of_neg+1)*batchsize, 1], each element is sqrt(sum(xi^2))
            decoder_norm = torch.sqrt(torch.sum(torch.square(decoder_output_rot), 1, True))
            #decoder_norm: [(num_of_neg+1)*batchsize, 1], each element is sqrt(sum(yi^2))
            encoder_output_rot = encoder_output.repeat(num_of_neg+1, 1) 
            encoder_decoder_prod = torch.sum(encoder_output_rot * decoder_output_rot, 1, True)
            #encoder_decoder_prod: [(num_of_neg+1)*batchsize, 1], each elemant is sum(xi*yi)
            encoder_decoder_prod_norm = encoder_norm * decoder_norm
            #encoder_decoder_prod_norm: [(num_of_neg+1)*batchsize, 1], each element is (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            cos_sim_raw = torch.true_divide(encoder_decoder_prod, encoder_decoder_prod_norm + 0.00001)
            #cos_sim_raw: [(num_of_neg+1)*batchsize, 1], each element is sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            cos_sim = torch.transpose(torch.reshape(torch.transpose(cos_sim_raw, 1, 0), [num_of_neg +1, encoder_output.shape[0]]), 1, 0)
            #cos_sim: [batchsize, num_of_neg+1], each element is 20 * sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            #first column is 0~batchsize-1, other columns are rotated
                
            #cos_sim: [batchsize, num_of_neg+1], each element is 20 * sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            #first column is 0~batchsize-1, other columns are rotated
                
            yesvalue = cos_sim [:, 0:1]
            sp = yesvalue.repeat(1, num_of_neg) 
            #yesvalue: [batchsize, 1]: this is the prob of the positive pair
            sn = cos_sim[0:, 1:]
            ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
            an = torch.clamp_min(sn.detach() + self.m, min=0.)

            delta_p = 1 - self.m
            delta_n = self.m

            logit_p = - ap * (sp - delta_p) * self.gamma
            logit_n = an * (sn - delta_n) * self.gamma

            loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
            return loss

        loss_a_to_b = calculator_circle_loss(encoder_output, decoder_output)
        loss_b_to_a = calculator_circle_loss(decoder_output, encoder_output)

        loss = (loss_a_to_b + loss_b_to_a) / 2.0

        loss = torch.mean(loss)

        return loss


class NegativeCircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(NegativeCircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, encoder_output: Tensor, decoder_output: Tensor) -> Tensor:
        num_of_neg = encoder_output.shape[0] -1
        def calculator_circle_loss(encoder_output, decoder_output):
            decoder_output_rot = decoder_output.repeat(1,1)
            for i in range(num_of_neg):
                decoder_output_rot = torch.cat((
                    decoder_output_rot, 
                    decoder_output[i+1:, : ],
                    decoder_output[0: i+1, : ],
                    ), 0)
            #decoder_output_rot: [(num_of_neg+1)*batchsize, dim]
            encoder_norm = torch.sqrt(torch.sum(torch.square(encoder_output), 1, True))
            encoder_norm = encoder_norm.repeat(num_of_neg+1, 1)
            #encoder_norm: [(num_of_neg+1)*batchsize, 1], each element is sqrt(sum(xi^2))
            decoder_norm = torch.sqrt(torch.sum(torch.square(decoder_output_rot), 1, True))
            #decoder_norm: [(num_of_neg+1)*batchsize, 1], each element is sqrt(sum(yi^2))
            encoder_output_rot = encoder_output.repeat(num_of_neg+1, 1) 
            encoder_decoder_prod = torch.sum(encoder_output_rot * decoder_output_rot, 1, True)
            #encoder_decoder_prod: [(num_of_neg+1)*batchsize, 1], each elemant is sum(xi*yi)
            encoder_decoder_prod_norm = encoder_norm * decoder_norm
            #encoder_decoder_prod_norm: [(num_of_neg+1)*batchsize, 1], each element is (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            cos_sim_raw = torch.true_divide(encoder_decoder_prod, encoder_decoder_prod_norm + 0.00001)
            #cos_sim_raw: [(num_of_neg+1)*batchsize, 1], each element is sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            cos_sim = torch.transpose(torch.reshape(torch.transpose(cos_sim_raw, 1, 0), [num_of_neg +1, encoder_output.shape[0]]), 1, 0)
            #cos_sim: [batchsize, num_of_neg+1], each element is 20 * sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            #first column is 0~batchsize-1, other columns are rotated
                
            #cos_sim: [batchsize, num_of_neg+1], each element is 20 * sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            #first column is 0~batchsize-1, other columns are rotated
                
            #yesvalue: [batchsize, 1]: this is the prob of the positive pair
            sn = cos_sim[0:, 1:]
            an = torch.clamp_min(sn.detach() + self.m, min=0.)

            delta_n = self.m

            logit_n = an * (sn - delta_n) * self.gamma

            loss = self.soft_plus(torch.logsumexp(logit_n, dim=0))
            return loss

        loss_a_to_b = calculator_circle_loss(encoder_output, decoder_output)
        loss_b_to_a = calculator_circle_loss(decoder_output, encoder_output)

        loss = (loss_a_to_b + loss_b_to_a) / 2.0

        loss = torch.mean(loss)

        return loss


loss_function_map = {
    "bcewithneg": CosineEmbeddingBCELossWithNegtive,
    "bce": CosineEmbeddingBCELoss,
    "circle": CircleLoss,
}