class TestNet1(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.case = 3

        n_filters = 3
        self.conv = nn.Conv1d(5, n_filters, 2, stride=2, bias=False)
        self.combine_filters = nn.Linear(n_filters, 1, bias=False)

    def forward(self, input_):
        x = self.conv(input_)
        x = F.relu(x)

        # x = F.avg_pool1d(x, x.size()[2]).sum(dim=1)
        # x = x.sum(dim=2) / input_.size()[0]

        batch_size, feature_size, data_size = x.size()

        if self.case == 1:
            x = x.sum(dim=2).sum(dim=1) / (data_size ** 1/2)
        elif self.case == 2:
            x = x.sum(dim=2) / (data_size ** 1/2)
            x = self.combine_filters(x).squeeze()
        elif self.case == 3:
            x = x.sum(dim=2)
            x = self.combine_filters(x).squeeze()
        elif self.case == 4:
            x = x.sum(dim=2)
            x = torch.cat([x, Variable(torch.ones(batch_size, 1) * data_size)], 1)
            x = self.combine_filters(x).squeeze()

        # x = x.sum(dim=2)

        # x = self.combine_filters(x).squeeze()

        x = F.sigmoid(x)
        return x



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        n_filters = 3
        self.spatial_conv = nn.Conv1d(5, n_filters, 2, stride=2, bias=False)
        self.combine_convs = nn.Linear(n_filters, 1, bias=False)

    def forward(self, aa, adjacency):
        x = aa @ adjacency.transpose(0, 1)
        x = self.spatial_conv(x)
        x = x @ adjacency[::2, :]
        x = x.sum(dim=2) / adjacency.sum(dim=0)
        x = self.combine_convs(x)
        x = x.squeeze()
        return F.sigmoid(x)
