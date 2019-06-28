import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(args):
    return MCBN(args)


class MCBN(nn.Module):
    def __init__(self, config):
        super(MCBN, self).__init__()
        in_channels = config.in_channels
        filter_config = (64, 128)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 2):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i]))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i]))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 2):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 2):
            feat = self.decoders[i](feat, indices[1 - i], unpool_sizes[1 - i])
        output = self.classifier(feat)
        results = {'mean': output}

        return results

class CustomBN(nn.Module):
    def __init__(self, n_in_feat):
        super(CustomBN, self).__init__()
        self.n_in_feat = n_in_feat
        self.bn2d = nn.BatchNorm2d(n_in_feat)

    def forward(self, x):
        if self.training:
            y = self.bn2d(x)
        else:
            x_size = x.size()
            half_batch = x_size[0]//2
            self.train()
            self.bn2d.running_mean = torch.zeros(self.n_in_feat)
            self.bn2d.running_var = torch.ones(self.n_in_feat)
            _ = self.bn2d(x[half_batch:])
            self.eval()
            y = self.bn2d(x)
        return y


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  CustomBN(n_out_feat),
                  nn.ReLU()]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       CustomBN(n_out_feat),
                       nn.ReLU()]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        super(_Decoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1),
                  CustomBN(n_in_feat),
                  nn.ReLU()]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                       CustomBN(n_out_feat),
                       nn.ReLU()]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)
