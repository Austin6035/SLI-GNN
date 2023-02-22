import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn

class Message_Passing_Layer(torch.nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_nbr):
        super(Message_Passing_Layer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.num_nbr = num_nbr
        self.bn1 = nn.BatchNorm1d(self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.bn3 = nn.BatchNorm1d(self.nbr_fea_len)
        self.bn4 = nn.BatchNorm1d(self.atom_fea_len)

        self.fc_core = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, self.atom_fea_len)
        self.fc_filter = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, self.atom_fea_len)
        self.fc_bond = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, self.nbr_fea_len)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, ):
        atom_nbr_fea = x[edge_index[1], :]
        atom_init_fea = x[edge_index[0], :]
        Z = torch.cat((atom_nbr_fea, atom_init_fea, edge_attr), dim=1)

        a_filter = self.bn1(self.fc_core(Z))
        a_core = self.bn2(self.fc_filter(Z))
        bond = self.bn3(self.fc_bond(Z))

        a_filter = torch.sigmoid(a_filter)
        a_core = F.softplus(a_core)

        nbr_sumed = a_filter * a_core
        nbr_sumed = nbr_sumed.reshape((-1, self.num_nbr, self.atom_fea_len))
        nbr_sumed = torch.sum(nbr_sumed, dim=1)
        nbr_sumed = self.bn4(nbr_sumed)
        out = F.softplus(x + nbr_sumed)
        bond_out = F.softplus(edge_attr + bond)
        return out, bond_out

class Dynamic_Atom_Embedding(torch.nn.Module):

    def __init__(self, emb_dim, properties_list=None):
        super(Dynamic_Atom_Embedding, self).__init__()
        self.properties_list = properties_list
        self.properties_name = ['N', 'G', 'P', 'NV', 'E', 'R', 'V', 'EA', 'I']
        self.full_dims = [100, 18, 7, 12, 10, 10, 10, 10, 10]
        full_atom_feature_dims = self.__get_full_dims()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            # adjust
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding

    def __get_full_dims(self):
        feature_dim = []
        if self.properties_list == 'all':
            feature_dim = self.full_dims
        elif len(self.properties_list) == 1 or self.properties_list[0] == 'N':
            feature_dim = [100]
        else:
            for prop in self.properties_list:
                index = self.properties_name.index(prop)
                feature_dim.append(self.full_dims[index])
        return feature_dim


class Net(torch.nn.Module):

    def __init__(self, orig_bond_fea_len=51, nbr_fea_len=128, atom_fea_len=64, n_conv=3, h_fea_len=128, l1=1,
                 l2=1, classification=False, n_classes=2, attention=False, dynamic_attention=False, n_heads=1,
                 max_num_nbr=12, pooling='mean', p=0, properties_list=None, atom_ref=None):
        super(Net, self).__init__()
        self.classification = classification
        self.pooling = pooling
        self.bn = nn.BatchNorm1d(atom_fea_len)
        self.atom_embedding = Dynamic_Atom_Embedding(atom_fea_len, properties_list=properties_list)
        self.bond_embedding = nn.Embedding(orig_bond_fea_len, nbr_fea_len)

        # U0参考值
        if atom_ref is not None:
            self.atomref_layer = nn.Embedding.from_pretrained(
                torch.from_numpy(atom_ref.astype(np.float32))
            )
        else:
            self.atomref_layer = None

        self.p = p
        if attention:
            self.n_gats = nn.ModuleList([torch_geometric.nn.GATConv(atom_fea_len, atom_fea_len,
                                                                     edge_dim=nbr_fea_len, heads=n_heads, concat=False)
                                          for _ in range(n_conv)])
        elif dynamic_attention:
            self.n_gats = nn.ModuleList([torch_geometric.nn.GATv2Conv(atom_fea_len, atom_fea_len,
                                                                       edge_dim=nbr_fea_len, heads=n_heads,
                                                                       concat=False)
                                          for _ in range(n_conv)])
        else:
            self.n_mpnns = nn.ModuleList(
                [Message_Passing_Layer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len, num_nbr=max_num_nbr)
                 for _ in range(n_conv)])

        if l1 > 0:
            self.conv_to_fc1 = nn.Linear(atom_fea_len * (n_conv + 1), h_fea_len)
            self.l1 = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                     for _ in range(l1 - 1)])

        if l2 > 0:
            if l1 == 0:
                self.conv_to_fc2 = nn.Linear(atom_fea_len * (n_conv + 1), h_fea_len)
            self.l2 = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                     for _ in range(l2)])

        if self.p > 0:
            self.dropout = nn.Dropout(p=p)
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, n_classes)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, data):
        x, edge_index, edge_weight, y = data.x, data.edge_index, data.edge_attr, data.y
        batch = data.batch

        if self.atomref_layer is not None:
            atomic_numbers = x.squeeze(1)
            energies = self.atomref_layer(atomic_numbers)
            energies = torch_geometric.nn.global_add_pool(energies, batch)

        x = self.atom_embedding(x)
        edge_weight = self.bond_embedding(edge_weight)
        info = x
        if hasattr(self, 'n_gats'):
            for conv in self.n_gats:
                x = conv(x=x, edge_index=edge_index, edge_attr=edge_weight)
                info = torch.cat((info, x), dim=1)
        if hasattr(self, 'n_mpnns'):
            for conv in self.n_mpnns:
                x, edge_weight = conv(x=x, edge_index=edge_index, edge_attr=edge_weight)
                info = torch.cat((info, x), dim=1)

        x = info
        if hasattr(self, 'conv_to_fc1'):
            x = F.softplus(self.conv_to_fc1(x))
        if hasattr(self, 'l1'):
            for hidden in self.l1:
                x = F.softplus(hidden(x))

        if self.pooling == 'add':
            x = torch_geometric.nn.global_add_pool(x, batch)
        elif self.pooling == 'max':
            x = torch_geometric.nn.global_max_pool(x, batch)
        else:
            x = torch_geometric.nn.global_mean_pool(x, batch)


        if hasattr(self, 'conv_to_fc2'):
            x = F.softplus(self.conv_to_fc2(x))
        if hasattr(self, 'l2'):
            for hidden in self.l2:
                x = F.softplus(hidden(x))


        x = self.fc_out(x)

        if self.p > 0:
            x = self.dropout(x)
        if self.atomref_layer is not None:
            x = x + energies
        return x
