import torch
from torch import nn
from torch_geometric.utils import add_remaining_self_loops, degree


class MyGCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        n = x.size(0)

        # Добавляем недостающие петли
        edge_index, _ = add_remaining_self_loops(
            edge_index, num_nodes=n
        )
        row, col = edge_index

        # Степени после добавления петель
        deg = degree(col, n, dtype=x.dtype).clamp(min=1)

        # Преобразуем признаки вершин
        h = self.lin(x)

        # Нормируем сообщения по степеням обоих концов ребра
        norm = deg[row].pow(-0.5) * deg[col].pow(-0.5)
        msg = h[row] * norm.unsqueeze(-1)

        # Складываем сообщения у получателей
        out = torch.zeros_like(h)
        out.index_add_(0, col, msg)

        return out

class MySAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index

        # Суммируем признаки входящих соседей
        summed = torch.zeros_like(x)
        summed.index_add_(0, col, x[row])

        # Получаем среднее, для вершин без соседей останется ноль
        deg = degree(col, x.size(0), dtype=x.dtype)
        mean_neighbors = summed / deg.clamp(min=1).unsqueeze(-1)

        # Отдельно преобразуем соседей и собственные признаки
        return self.lin_neigh(mean_neighbors) + self.lin_self(x)
