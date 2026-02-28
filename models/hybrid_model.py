import torch
import torch.nn as nn

# --- Basic ConvLSTM Cell ---
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# --- CNN Encoder ---
class CNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CNNEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

# --- Main Hybrid Model Class ---
class HybridCNNConvLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, kernel_size=(3,3), num_layers=1, output_dim=1):
        super(HybridCNNConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.spatial_encoder = CNNEncoder(input_dim, hidden_dim)
        
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.hidden_dim 
            self.cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dim, kernel_size, bias=True))
            
        self.final_conv = nn.Conv2d(self.hidden_dim, output_dim, kernel_size=1)

    def forward(self, x, future_steps=6):
        b, seq_len, c, h, w = x.size()
        
        # 1. Spatial Encoding
        cnn_in = x.view(b * seq_len, c, h, w) 
        cnn_out = self.spatial_encoder(cnn_in)
        cnn_out = cnn_out.view(b, seq_len, self.hidden_dim, h, w)
        
        # 2. Temporal Encoding
        hidden_state = [cell.init_hidden(b, (h, w)) for cell in self.cell_list]
        current_input = cnn_out
        
        for t in range(seq_len):
            layer_input = current_input[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                h_next, c_next = self.cell_list[layer_idx](layer_input, (h, c))
                hidden_state[layer_idx] = (h_next, c_next)
                layer_input = h_next 

        # 3. Decoding
        outputs = []
        last_hidden = hidden_state[-1][0]
        for _ in range(future_steps):
            pred = self.final_conv(last_hidden)
            outputs.append(pred)
            
        return torch.stack(outputs, dim=1)
