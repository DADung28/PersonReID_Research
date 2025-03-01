from collections import defaultdict

import numpy as np
import torch

from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.losses import TripletMarginLoss


def concat_indices_tuple(x):
    return [torch.cat(y) for y in zip(*x)]


class CentroidTripletLoss(BaseMetricLossFunction):
    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.triplet_loss = TripletMarginLoss(
            margin=margin,
            swap=swap,
            smooth_loss=smooth_loss,
            triplets_per_anchor=triplets_per_anchor,
            **kwargs
        )

    def compute_loss(
        self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        c_f.indices_tuple_not_supported(indices_tuple)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        """
        "During training stage each mini-batch contains 𝑃 distinct item
        classes with 𝑀 samples per class, resulting in batch size of 𝑃 × 𝑀."
        """
        masks, class_masks, labels_list, query_indices = self.create_masks_train(labels)

        P = len(labels_list)
        M = max([len(instances) for instances in labels_list])
        DIM = embeddings.size(-1)

        """
        "...each sample from S𝑘 is used as a query 𝑞𝑘 and the rest 
        𝑀 −1 samples are used to build a prototype centroid"
        i.e. for each class k of M items, we make M pairs of (query, centroid),
        making a total of P*M total pairs.
        masks = (M*P x len(embeddings)) matrix
        labels_list[i] = indicies of embeddings belonging to ith class
        centroids_emd.shape == (M*P, DIM)
        i.e.    centroids_emb[0] == centroid vector for 0th class, where the first embedding is the query vector
                centroids_emb[1] == centroid vector for 0th class, where the second embedding is the query vector
                centroids_emb[M+1] == centroid vector for 1th class, where the first embedding is the query vector
        """

        masks_float = masks.type(embeddings.type()).to(embeddings.device)
        class_masks_float = class_masks.type(embeddings.type()).to(embeddings.device)
        inst_counts = masks_float.sum(-1)
        class_inst_counts = class_masks_float.sum(-1)

        valid_mask = inst_counts > 0
        padded = masks_float.unsqueeze(-1) * embeddings.unsqueeze(0)
        class_padded = class_masks_float.unsqueeze(-1) * embeddings.unsqueeze(0)

        positive_centroids_emb = padded.sum(-2) / inst_counts.masked_fill(
            inst_counts == 0, 1
        ).unsqueeze(-1)

        negative_centroids_emb = class_padded.sum(-2) / class_inst_counts.masked_fill(
            class_inst_counts == 0, 1
        ).unsqueeze(-1)

        query_indices = torch.tensor(query_indices).to(embeddings.device)
        query_embeddings = embeddings.index_select(0, query_indices)
        query_labels = labels.index_select(0, query_indices)
        assert positive_centroids_emb.size() == (M * P, DIM)
        assert negative_centroids_emb.size() == (P, DIM)
        assert query_embeddings.size() == (M * P, DIM)

        query_indices = query_indices.view((P, M)).transpose(0, 1)
        query_embeddings = query_embeddings.view((P, M, -1)).transpose(0, 1)
        query_labels = query_labels.view((P, M)).transpose(0, 1)
        positive_centroids_emb = positive_centroids_emb.view((P, M, -1)).transpose(0, 1)
        valid_mask = valid_mask.view((P, M)).transpose(0, 1)

        labels_collect = []
        embeddings_collect = []
        tuple_indices_collect = []
        starting_idx = 0
        for inst_idx in range(M):
            one_mask = valid_mask[inst_idx]
            if torch.sum(one_mask) > 1:
                anchors = query_embeddings[inst_idx][one_mask]
                pos_centroids = positive_centroids_emb[inst_idx][one_mask]
                one_labels = query_labels[inst_idx][one_mask]

                embeddings_concat = torch.cat(
                    (anchors, pos_centroids, negative_centroids_emb)
                )
                labels_concat = torch.cat(
                    (one_labels, one_labels, query_labels[inst_idx])
                )
                indices_tuple = get_all_triplets_indices(labels_concat)

                """
                Right now indices tuple considers all embeddings in
                embeddings_concat as anchors, pos_example, neg_examples.
                1. make only query vectors be anchor vectors
                2. make pos_centroids be only used as a positive example
                3. negative as so
                """
                # make only query vectors be anchor vectors
                indices_tuple = [x[: len(x) // 3] + starting_idx for x in indices_tuple]

                # make only pos_centroids be postive examples
                indices_tuple = [x.view(len(one_labels), -1) for x in indices_tuple]
                indices_tuple = [x.chunk(2, dim=1)[0] for x in indices_tuple]

                # make only neg_centroids be negative examples
                indices_tuple = [
                    x.chunk(len(one_labels), dim=1)[-1].flatten() for x in indices_tuple
                ]

                tuple_indices_collect.append(indices_tuple)
                embeddings_collect.append(embeddings_concat)
                labels_collect.append(labels_concat)
                starting_idx += len(labels_concat)

        indices_tuple = concat_indices_tuple(tuple_indices_collect)

        if len(indices_tuple) == 0:
            return self.zero_losses()

        final_embeddings = torch.cat(embeddings_collect)
        final_labels = torch.cat(labels_collect)

        loss = self.triplet_loss.compute_loss(
            final_embeddings, final_labels, indices_tuple, ref_emb=None, ref_labels=None
        )
        return loss
    
    def create_masks_train(self, class_labels):
        labels_dict = defaultdict(list)
        class_labels = class_labels.detach().cpu().numpy()
        for idx, pid in enumerate(class_labels):
            labels_dict[pid].append(idx)

        unique_classes = list(labels_dict.keys())
        labels_list = list(labels_dict.values())
        lens_list = [len(item) for item in labels_list]
        lens_list_cs = np.cumsum(lens_list)

        M = max(len(instances) for instances in labels_list)
        P = len(unique_classes)

        query_indices = []
        class_masks = torch.zeros((P, len(class_labels)), dtype=bool)
        masks = torch.zeros((M * P, len(class_labels)), dtype=bool)
        for class_idx, class_insts in enumerate(labels_list):
            class_masks[class_idx, class_insts] = 1
            for instance_idx in range(M):
                matrix_idx = class_idx * M + instance_idx
                if instance_idx < len(class_insts):
                    query_indices.append(class_insts[instance_idx])
                    ones = class_insts[:instance_idx] + class_insts[instance_idx + 1 :]
                    masks[matrix_idx, ones] = 1
                else:
                    query_indices.append(class_insts[0])
        return masks, class_masks, labels_list, query_indices


def get_all_triplets_indices(labels, ref_labels=None):
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)


def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs
