# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TubeDETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import RobertaModel, RobertaTokenizerFast
import math

from .position_encoding import TimeEmbeddingSine, TimeEmbeddingLearned


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        video_max_len=0,
        stride=0,
        no_tsa=False,
        return_weights=False,
        fast=False,
        fast_mode="",
        learn_time_embed=False,
        rd_init_tsa=False,
        no_time_embed=False,
    ):
        """
        :param d_model: transformer embedding dimension
        :param nhead: transformer number of heads
        :param num_encoder_layers: transformer encoder number of layers
        :param num_decoder_layers: transformer decoder number of layers
        :param dim_feedforward: transformer dimension of feedforward
        :param dropout: transformer dropout
        :param activation: transformer activation
        :param return_intermediate_dec: whether to return intermediate outputs of the decoder
        :param pass_pos_and_query: if True tgt is initialized to 0 and position is added at every layer
        :param text_encoder_type: Hugging Face name for the text encoder
        :param freeze_text_encoder: whether to freeze text encoder weights
        :param video_max_len: maximum number of frames in the model
        :param stride: temporal stride k
        :param no_tsa: whether to use temporal self-attention
        :param return_weights: whether to return attention weights
        :param fast: whether to use the fast branch
        :param fast_mode: which variant of fast branch to use
        :param learn_time_embed: whether to learn time encodings
        :param rd_init_tsa: whether to randomly initialize temporal self-attention weights
        :param no_time_embed: whether to use time encodings
        """
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm, return_weights=True
        )

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            no_tsa=no_tsa,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            return_weights=return_weights,
        )

        self._reset_parameters()

        self.return_weights = return_weights

        self.learn_time_embed = learn_time_embed
        self.use_time_embed = not no_time_embed
        if self.use_time_embed:
            if learn_time_embed:
                self.time_embed = TimeEmbeddingLearned(video_max_len, d_model)
            else:
                self.time_embed = TimeEmbeddingSine(video_max_len, d_model)

        self.fast = fast
        self.fast_mode = fast_mode
        if fast:
            if fast_mode == "gating":
                self.fast_encoder = nn.Linear(d_model, d_model)
            elif fast_mode == "transformer":
                encoder_layer = TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation
                )
                self.fast_encoder = TransformerEncoder(
                    encoder_layer, 1, nn.LayerNorm(d_model), return_weights=True
                )
                self.fast_residual = nn.Linear(d_model, d_model)
            else:
                self.fast_encoder = nn.Linear(d_model, d_model)
                self.fast_residual = nn.Linear(d_model, d_model)

        self.rd_init_tsa = rd_init_tsa
        self._reset_temporal_parameters()

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            text_encoder_type, local_files_only=True
        )
        self.text_encoder = RobertaModel.from_pretrained(
            text_encoder_type, local_files_only=True
        )

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.video_max_len = video_max_len
        self.stride = stride

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _reset_temporal_parameters(self):
        for n, p in self.named_parameters():
            if "fast_encoder" in n and self.fast_mode == "transformer":
                if "norm" in n and "weight" in n:
                    nn.init.constant_(p, 1.0)
                elif "norm" in n and "bias" in n:
                    nn.init.constant_(p, 0)
                else:
                    nn.init.constant_(p, 0)

            if self.rd_init_tsa and "decoder" in n and "self_attn" in n:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            if "fast_residual" in n:
                nn.init.constant_(p, 0)
            if self.fast_mode == "gating" and "fast_encoder" in n:
                nn.init.constant_(p, 0)

    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        encode_and_save=True,
        durations=None,
        tpad_mask_t=None,
        fast_src=None,
        img_memory=None,
        query_mask=None,
        text_memory=None,
        text_mask=None,
        memory_mask=None,
    ):
        if encode_and_save:
            # flatten n_clipsxCxHxW to HWxn_clipsxC
            tot_clips, c, h, w = src.shape
            device = src.device

            # nb of times object queries are repeated
            if durations is not None:
                t = max(durations)
                b = len(durations)
                bs_oq = tot_clips if (not self.stride) else b * t
            else:
                bs_oq = tot_clips

            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)
            query_embed = query_embed.unsqueeze(1).repeat(
                1, bs_oq, 1
            )  # n_queriesx(b*t)xf

            n_queries, _, f = query_embed.shape
            query_embed = query_embed.view(
                n_queries * t,
                b,
                f,
            )
            if self.use_time_embed:  # add temporal encoding to init time queries
                time_embed = self.time_embed(t).repeat(n_queries, b, 1)
                query_embed = query_embed + time_embed

            # prepare time queries mask
            query_mask = None
            if self.stride:
                query_mask = (
                    torch.ones(
                        b,
                        n_queries * t,
                    )
                    .bool()
                    .to(device)
                )
                query_mask[:, 0] = False  # avoid empty masks
                for i_dur, dur in enumerate(durations):
                    query_mask[i_dur, : (dur * n_queries)] = False

            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = (
                    src + 0.1 * pos_embed,
                    query_embed,
                    None,
                    None,
                )

            if isinstance(text[0], str):
                # Encode the text
                tokenized = self.tokenizer.batch_encode_plus(
                    text, padding="longest", return_tensors="pt"
                ).to(device)
                encoded_text = self.text_encoder(**tokenized)

                # Transpose memory because pytorch's attention expects sequence first
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                text_memory_resized = self.resizer(text_memory)
            else:
                # The text is already encoded, use as is.
                text_attention_mask, text_memory_resized, tokenized = text

            # encode caption once per video and repeat each caption X times where X is the number of clips in the video
            n_repeat = t if (not self.stride) else math.ceil(t / self.stride)
            assert (
                n_repeat
                == src.shape[1] // text_memory_resized.shape[1]
                == mask.shape[0] // text_attention_mask.shape[0]
            )
            tokenized._encodings = [
                elt for elt in tokenized._encodings for _ in range(n_repeat)
            ]  # repeat batchencodings output (BT)
            text_attention_mask_orig = text_attention_mask
            text_attention_mask = torch.stack(
                [
                    text_attention_mask[i_elt]
                    for i_elt in range(len(text_attention_mask))
                    for _ in range(n_repeat)
                ]
            )
            text_memory_resized_orig = text_memory_resized
            text_memory_resized = torch.stack(
                [
                    text_memory_resized[:, i_elt]
                    for i_elt in range(text_memory_resized.size(1))
                    for _ in range(n_repeat)
                ],
                1,
            )
            tokenized["input_ids"] = torch.stack(
                [
                    tokenized["input_ids"][i_elt]
                    for i_elt in range(len(tokenized["input_ids"]))
                    for _ in range(n_repeat)
                ]
            )
            tokenized["attention_mask"] = torch.stack(
                [
                    tokenized["attention_mask"][i_elt]
                    for i_elt in range(len(tokenized["attention_mask"]))
                    for _ in range(n_repeat)
                ]
            )

            # Concat on the sequence dimension
            src = torch.cat([src, text_memory_resized], dim=0)

            # Concat mask for all frames, will be used for the decoding
            if tpad_mask_t is not None:
                tpad_mask_t_orig = tpad_mask_t
                tpad_mask_t = tpad_mask_t.flatten(1)  # bxtxhxw -> bx(txhxw)
                text_attn_mask_t = torch.stack(
                    [
                        text_attention_mask_orig[i_elt]
                        for i_elt in range(len(text_attention_mask_orig))
                        for _ in range(max(durations))
                    ]
                )
                tpad_mask_t = torch.cat([tpad_mask_t, text_attn_mask_t], dim=1)

            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_attention_mask], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat(
                [pos_embed, torch.zeros_like(text_memory_resized)], dim=0
            )

            if (
                self.fast_mode == "noslow"
            ):  # no space-text attention for noslow baseline
                img_memory, weights = src, None
                text_memory = torch.stack(
                    [
                        text_memory_resized_orig[:, i_elt]
                        for i_elt in range(text_memory_resized_orig.size(1))
                        for _ in range(t)
                    ],
                    1,
                )
            else:  # space-text attention
                img_memory, weights = self.encoder(
                    src, src_key_padding_mask=mask, pos=pos_embed, mask=None
                )
                text_memory = img_memory[-len(text_memory_resized) :]

            if self.fast:
                if (
                    self.fast_mode == "transformer"
                ):  # temporal transformer in the fast branch for this variant
                    fast_src2 = (
                        fast_src.flatten(2)
                        .view(b, t, f, h * w)
                        .permute(1, 0, 3, 2)
                        .flatten(1, 2)
                    )  # (b*t)xfxhxw -> (b*t)xfx(h*w) -> bxtxfx(h*w) -> txbx(h*w)xf -> tx(b*h*w)xf
                    time_embed = self.time_embed(t)
                    time_embed = time_embed.repeat(1, b * h * w, 1)
                    fast_memory, fast_weights = self.fast_encoder(
                        fast_src2, pos=time_embed
                    )
                    fast_memory = (
                        fast_memory.view(t, b, h * w, f)
                        .transpose(0, 1)
                        .view(b * t, h * w, f)
                        .transpose(0, 1)
                    )  # tx(b*h*w)xf -> txbx(h*w)xf -> bxtx(h*w)xf -> (b*t)x(h*w)xf -> (h*w)x(b*t)xf
                else:
                    fast_src2 = fast_src.flatten(2).permute(
                        2, 0, 1
                    )  # (b*t)xfxhxw -> (b*t)xfx(h*w) -> (h*w)x(b*t)xf
                    if (
                        self.fast_mode == "pool"
                    ):  # spatial pool in the fast branch for this baseline
                        fast_mask = tpad_mask_t_orig.flatten(1).transpose(
                            0, 1
                        )  # (h*w)x(b*t)
                        fast_pool_mask = ~fast_mask[:, :, None]
                        sum_mask = fast_pool_mask.float().sum(dim=0).clamp(min=1)
                        fast_src2 = fast_src2 * fast_pool_mask
                        n_visual_tokens = len(fast_src2)
                        fast_src2 = torch.div(fast_src2.sum(dim=0), sum_mask)
                    fast_memory = self.fast_encoder(fast_src2)
                    if self.fast_mode == "pool":
                        fast_memory = fast_memory.unsqueeze(0).repeat(
                            n_visual_tokens, 1, 1
                        )

            if self.stride:  # temporal replication
                device = img_memory.device
                n_tokens, tot_clips, f = img_memory.shape
                pad_img_memory = torch.zeros(n_tokens, b, t, f).to(device)
                pad_pos_embed = torch.zeros(n_tokens, b, t, f).to(device)
                cur_clip = 0
                n_clips = math.ceil(t / self.stride)
                for i_dur, dur in enumerate(durations):
                    for i_clip in range(n_clips):
                        clip_dur = min(self.stride, t - i_clip * self.stride)
                        pad_img_memory[
                            :,
                            i_dur,
                            i_clip * self.stride : i_clip * self.stride + clip_dur,
                        ] = (
                            img_memory[:, cur_clip].unsqueeze(1).repeat(1, clip_dur, 1)
                        )
                        pad_pos_embed[
                            :,
                            i_dur,
                            i_clip * self.stride : i_clip * self.stride + clip_dur,
                        ] = (
                            pos_embed[:, cur_clip].unsqueeze(1).repeat(1, clip_dur, 1)
                        )
                        cur_clip += 1
                img_memory = pad_img_memory.view(
                    n_tokens, b * t, f
                )  # n_tokensxbxtxf -> n_tokensx(b*t)xf
                mask = tpad_mask_t.view(
                    b * t, n_tokens
                )  # bxtxn_tokens -> (b*t)xn_tokens
                mask[:, 0] = False  # avoid empty masks
                pos_embed = pad_pos_embed.view(
                    n_tokens, b * t, f
                )  # n_tokensxbxtxf -> n_tokensx(b*t)xf

                if self.fast:  # aggregate slow and fast features
                    n_visual_tokens = len(fast_memory)
                    if self.fast_mode == "noslow":
                        img_memory = torch.cat([fast_memory, text_memory], 0)
                    elif self.fast_mode == "gating":
                        img_memory2 = img_memory[
                            :n_visual_tokens
                        ].clone() * torch.sigmoid(fast_memory)
                        img_memory[:n_visual_tokens] = (
                            img_memory[:n_visual_tokens] + img_memory2
                        )
                    else:
                        img_memory2 = img_memory[:n_visual_tokens] + fast_memory
                        img_memory2 = self.fast_residual(img_memory2)
                        img_memory[:n_visual_tokens] = (
                            img_memory[:n_visual_tokens] + img_memory2
                        )
                text_memory = img_memory[-len(text_memory_resized) :]

            memory_cache = {
                "text_memory_resized": text_memory_resized,  # seq first
                "text_memory": text_memory,  # seq first
                "text_attention_mask": text_attention_mask,  # batch first
                "tokenized": tokenized,  # batch first
                "img_memory": img_memory,  # seq first
                "mask": mask,  # batch first
                "pos_embed": pos_embed,  # seq first
                "query_embed": query_embed,  # seq first
                "query_mask": query_mask,  # batch first
            }

            return memory_cache

        else:
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = (
                    src + 0.1 * pos_embed,
                    query_embed,
                    None,
                    None,
                )

            # time-space-text attention
            hs = self.decoder(
                tgt,  # n_queriesx(b*t)xF
                img_memory,  # ntokensx(b*t)x256
                memory_key_padding_mask=mask,  # (b*t)xn_tokens
                pos=pos_embed,  # n_tokensx(b*t)xF
                query_pos=query_embed,  # n_queriesx(b*t)xF
                tgt_key_padding_mask=query_mask,  # bx(t*n_queries)
                text_memory=text_memory,
                text_memory_mask=text_mask,
                memory_mask=memory_mask,
            )  # n_layersxn_queriesx(b*t)xF
            if self.return_weights:
                hs, weights, cross_weights = hs

            if not self.return_weights:
                return hs.transpose(1, 2)
            else:
                return hs.transpose(1, 2), weights, cross_weights


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, return_weights=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_weights = return_weights

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src

        weights = None if not self.return_weights else []
        for layer in self.layers:
            output, cur_weights = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
            if self.return_weights:
                weights.append(cur_weights)

        if self.norm is not None:
            output = self.norm(output)

        if not self.return_weights:
            return output
        else:
            return output, torch.stack(weights)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        return_weights=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.return_weights = return_weights

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        text_memory=None,
        text_memory_mask=None,
    ):
        output = tgt

        intermediate = []
        intermediate_weights = []
        intermediate_cross_weights = []

        for i_layer, layer in enumerate(self.layers):
            output, weights, cross_weights = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                text_memory=text_memory,
                text_memory_mask=text_memory_mask,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.return_weights:
                    intermediate_weights.append(weights)
                    intermediate_cross_weights.append(cross_weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if not self.return_weights:
                return torch.stack(intermediate)
            else:
                return (
                    torch.stack(intermediate),
                    torch.stack(intermediate_weights),
                    torch.stack(intermediate_cross_weights),
                )

        if not self.return_weights:
            return output
        else:
            return output, weights, cross_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        q = k = self.with_pos_embed(src, pos)
        src2, weights = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        no_tsa=False,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.no_tsa = no_tsa

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        text_memory=None,
        text_memory_mask=None,
    ):

        q = k = self.with_pos_embed(tgt, query_pos)

        # Temporal Self attention
        if self.no_tsa:
            t, b, _ = tgt.shape
            n_tokens, bs, f = memory.shape
            tgt2, weights = self.self_attn(
                q.transpose(0, 1).reshape(bs * b, -1, f).transpose(0, 1),
                k.transpose(0, 1).reshape(bs * b, -1, f).transpose(0, 1),
                value=tgt.transpose(0, 1).reshape(bs * b, -1, f).transpose(0, 1),
                attn_mask=tgt_mask,
                key_padding_mask=None,
            )
            tgt2 = tgt2.view(b, t, f).transpose(0, 1)
        else:
            tgt2, weights = self.self_attn(
                q,
                k,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Time Aligned Cross attention
        t, b, _ = tgt.shape
        n_tokens, bs, f = memory.shape
        tgt_cross = (
            tgt.transpose(0, 1).reshape(bs, -1, f).transpose(0, 1)
        )  # txbxf -> bxtxf -> (b*t)x1xf -> 1x(b*t)xf
        query_pos_cross = (
            query_pos.transpose(0, 1).reshape(bs, -1, f).transpose(0, 1)
        )  # txbxf -> bxtxf -> (b*t)x1xf -> 1x(b*t)xf

        tgt2, cross_weights = self.cross_attn_image(
            query=self.with_pos_embed(tgt_cross, query_pos_cross),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )

        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, weights, cross_weights


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
        video_max_len=args.video_max_len_train,
        stride=args.stride,
        no_tsa=args.no_tsa,
        return_weights=args.guided_attn,
        fast=args.fast,
        fast_mode=args.fast_mode,
        learn_time_embed=args.learn_time_embed,
        rd_init_tsa=args.rd_init_tsa,
        no_time_embed=args.no_time_embed,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
