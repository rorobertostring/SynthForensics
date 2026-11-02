from .base_options import BaseOption


class TrainOption(BaseOption):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('-c', '--classes', type=str, nargs='+', default=['youtube', 'stablevideodiffusion'], help='the dataset for training')
        self.parser.add_argument('--fix-split', action='store_true', help='use the fixed split file for training validation split')
        self.parser.add_argument('--split-path', type=str, default='./splits', help='the split directory')
        self.parser.add_argument('--split-ratio', type=int, nargs='+', default=[0.9, 0.1], help='the split ratio for train and validation if split is not specified')
        self.parser.add_argument('--bs', type=int, default=2, help='batch size for training')
        self.parser.add_argument('--mode', type=str, default='train', help='mode')
        self.parser.add_argument('--epoch', type=int, default=50, help='epoch')
        self.parser.add_argument('--lr', type=float, default=2e-5, help='lr')
        self.parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay')
        self.parser.add_argument('--patience', type=int, default=4, help='patience')
        self.parser.add_argument('--cooldown', type=int, default=5, help='cool down')
        self.parser.add_argument('--step-factor', type=float, default=0.2, help='step factor for lr adjustment')
        self.parser.add_argument('--num-workers', type=int, default=4, help='number for dataloader workers per gpu')
        
        self.parser.add_argument('--log-dir', type=str, default='log_dir', help='log dirs')
        self.parser.add_argument('--save-epoch', type=int, default=5, help='interval for each saving')
        self.parser.add_argument('--display-step', type=int, default=50, help='display steps')
        self.parser.add_argument('--log-step', type=int, default=100, help='log steps')
        self.parser.add_argument('--val-step', type=int, default=1000, help='validation steps')
        self.parser.add_argument('--val-bound', type=float, default=1e-11, help='the minimum value for validation metric to finish training')
