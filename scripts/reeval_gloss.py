import torch
from config import parse_arguments
from utils import load_data, prepare_dataloaders, setup_logger
from model import BertClassifier
from training import test_evaluate
import sys
sys.argv = ['x', '--dataset', 'R8', '--bert_init', 'distilbert-base-uncased', '--batch_size', '16', '--max_length', '128']

args = parse_arguments()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_dir = './supervised_checkpoint/gloss_distilbert-base-uncased_R8_root_2026-10-25-14-26-01'

train, val, test = load_data('../data/R8/')
_, _, test_loader, num_labels = prepare_dataloaders(args, train, val, test)
model = BertClassifier(pretrained_model=args.bert_init, nb_class=num_labels).to(device)

ckpt = torch.load(f'{ckpt_dir}/checkpoint.pth')
model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded epoch {ckpt['epoch']+1} (should be epoch 4 — that's where val F1 peaked at 92.43)")

logger = setup_logger(ckpt_dir)
test_evaluate(model, test_loader, logger, device)
