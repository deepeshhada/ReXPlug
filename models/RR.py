import torch
import torch.nn as nn


class ReviewRegularizer(nn.Module):
    def __init__(self, num_factors):
        super(ReviewRegularizer, self).__init__()
        input_size = num_factors * 8
        self.model = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(input_size, 512),
            nn.Sigmoid()
        )

    def forward(self, interaction):
        output = self.model(interaction)
        return output


class RatingPredictor(nn.Module):
    def __init__(self, review_regularizer, num_users, num_items, num_factors=16, num_layers=4):
        super(NCF, self).__init__()

        embed_dim = num_factors * (2 ** (num_layers - 1))
        self.embed_user_MLP = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.embed_item_MLP = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        self.dropout = nn.Dropout(p=0.5)
        MLP_modules = []
        for i in range(num_layers):
            input_size = num_factors * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=0.4))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_forward = nn.Sequential(*MLP_modules)
        predict_size = num_factors
        self.predict = nn.Linear(predict_size, 1)

        self.review_regularizer = review_regularizer

    def forward(self, user, item, review=None):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_forward(self.dropout(interaction))
        preds = [self.predict(output_MLP).view(-1)]

        if review is not None:
            regularizer = self.review_regularizer(interaction)
            preds.append(regularizer)

        return preds
