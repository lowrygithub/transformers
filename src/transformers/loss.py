import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple

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
        negsample = encoder_output.shape[0] -1
        def calculator_circle_loss(encoder_output, decoder_output):
            decoder_output_rot = decoder_output.repeat(1,1)
            for i in range(negsample):
                decoder_output_rot = torch.cat((
                    decoder_output_rot, 
                    decoder_output[i+1:, : ],
                    decoder_output[0: i+1, : ],
                    ), 0)
            #decoder_output_rot: [(negsample+1)*batchsize, dim]
            encoder_norm = torch.sqrt(torch.sum(torch.square(encoder_output), 1, True))
            encoder_norm = encoder_norm.repeat(negsample+1, 1)
            #encoder_norm: [(negsample+1)*batchsize, 1], each element is sqrt(sum(xi^2))
            decoder_norm = torch.sqrt(torch.sum(torch.square(decoder_output_rot), 1, True))
            #decoder_norm: [(negsample+1)*batchsize, 1], each element is sqrt(sum(yi^2))
            encoder_output_rot = encoder_output.repeat(negsample+1, 1) 
            encoder_decoder_prod = torch.sum(encoder_output_rot * decoder_output_rot, 1, True)
            #encoder_decoder_prod: [(negsample+1)*batchsize, 1], each elemant is sum(xi*yi)
            encoder_decoder_prod_norm = encoder_norm * decoder_norm
            #encoder_decoder_prod_norm: [(negsample+1)*batchsize, 1], each element is (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            cos_sim_raw = torch.true_divide(encoder_decoder_prod, encoder_decoder_prod_norm + 0.00001)
            #cos_sim_raw: [(negsample+1)*batchsize, 1], each element is sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            cos_sim = torch.transpose(torch.reshape(torch.transpose(cos_sim_raw, 1, 0), [negsample +1, encoder_output.shape[0]]), 1, 0)
            #cos_sim: [batchsize, negsample+1], each element is 20 * sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            #first column is 0~batchsize-1, other columns are rotated
                
            #cos_sim: [batchsize, negsample+1], each element is 20 * sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            #first column is 0~batchsize-1, other columns are rotated
                
            yesvalue = cos_sim [:, 0:1]
            sp = yesvalue.repeat(1, negsample) 
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
        negsample = encoder_output.shape[0] -1
        def calculator_circle_loss(encoder_output, decoder_output):
            decoder_output_rot = decoder_output.repeat(1,1)
            for i in range(negsample):
                decoder_output_rot = torch.cat((
                    decoder_output_rot, 
                    decoder_output[i+1:, : ],
                    decoder_output[0: i+1, : ],
                    ), 0)
            #decoder_output_rot: [(negsample+1)*batchsize, dim]
            encoder_norm = torch.sqrt(torch.sum(torch.square(encoder_output), 1, True))
            encoder_norm = encoder_norm.repeat(negsample+1, 1)
            #encoder_norm: [(negsample+1)*batchsize, 1], each element is sqrt(sum(xi^2))
            decoder_norm = torch.sqrt(torch.sum(torch.square(decoder_output_rot), 1, True))
            #decoder_norm: [(negsample+1)*batchsize, 1], each element is sqrt(sum(yi^2))
            encoder_output_rot = encoder_output.repeat(negsample+1, 1) 
            encoder_decoder_prod = torch.sum(encoder_output_rot * decoder_output_rot, 1, True)
            #encoder_decoder_prod: [(negsample+1)*batchsize, 1], each elemant is sum(xi*yi)
            encoder_decoder_prod_norm = encoder_norm * decoder_norm
            #encoder_decoder_prod_norm: [(negsample+1)*batchsize, 1], each element is (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            cos_sim_raw = torch.true_divide(encoder_decoder_prod, encoder_decoder_prod_norm + 0.00001)
            #cos_sim_raw: [(negsample+1)*batchsize, 1], each element is sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            cos_sim = torch.transpose(torch.reshape(torch.transpose(cos_sim_raw, 1, 0), [negsample +1, encoder_output.shape[0]]), 1, 0)
            #cos_sim: [batchsize, negsample+1], each element is 20 * sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
            #first column is 0~batchsize-1, other columns are rotated
                
            #cos_sim: [batchsize, negsample+1], each element is 20 * sum(xi*yi) / (sqrt(sum(xi^2))) * (sqrt(sum(yi^2)))
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


