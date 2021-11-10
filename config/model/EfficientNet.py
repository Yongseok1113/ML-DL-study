import torch
from config.module import Conv, Residual, initialize_weights, SiLU, SE


class EfficientNet(torch.nn.Module):
    def __init__(self, args, num_class=1000) -> None:
        super().__init__()
        gate_fn = [True, False]
        filters = [24, 48, 64, 128, 160, 272, 1792]
        feature = [Conv(args, 3, filters[0], torch.nn.SiLU(), 3, 2)]

        if args.tf:
            filters[5] = 256
            filters[6] = 1280

        for i in range(2):
            if i == 0:
                feature.append(Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))
            else:
                feature.append(Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(Residual(args, filters[0], filters[1], 2, 4, gate_fn[0]))
            else:
                feature.append((Residual(args, filters[1], filters[1], 1, 4, gate_fn[0])))

        for i in range(4):
            if i == 0:
                feature.append(Residual(args, filters[1], filters[2], 2, 4, gate_fn[0]))
            else:
                feature.append((Residual(args, filters[2], filters[2], 1, 4, gate_fn[0])))

        for i in range(6):
            if i == 0:
                feature.append(Residual(args, filters[2], filters[3], 2, 4, gate_fn[1]))
            else:
                feature.append((Residual(args, filters[3], filters[3], 1, 4, gate_fn[1])))

        for i in range(9):
            if i == 0:
                feature.append(Residual(args, filters[3], filters[4], 1, 6, gate_fn[1]))
            else:
                feature.append((Residual(args, filters[4], filters[4], 1, 6, gate_fn[1])))

        for i in range(15):
            if i == 0:
                feature.append(Residual(args, filters[4], filters[5], 2, 6, gate_fn[1]))
            else:
                feature.append((Residual(args, filters[5], filters[5], 1, 6, gate_fn[1])))

        feature.append(Conv(args, filters[5], filters[6], torch.nn.SiLU()))

        self.feature = torch.nn.Sequential(*feature)
        self.fc = torch.nn.Sequential(torch.nn.Dropout(0.3, True),
                                      torch.nn.Linear(filters[6], num_class))

        initialize_weights(self)

    def forward(self,  x):
        x = self.feature(x)
        return self.fc(x.mean((2, 3)))

    def export(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'silu'):
                if isinstance(m.silu, torch.nn.SiLU):
                    m.silu = SiLU()

            if type(m) is SE:
                if isinstance(m.se[1], torch.nn.SiLU):
                    m.se[1] = SiLU()

        return self


