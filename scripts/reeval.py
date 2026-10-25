import sys, torch
sys.argv = ['x', '--dataset', 'R8', '--bert_init', 'distilbert-base-uncased', '--batch_size', '16', '--max_length', '128']
from config import parse_arguments
from utils import load_data, prepare_dataloaders, setup_logger
from model import BertClassifier
from training import test_evaluate

args = parse_arguments()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_dir = './supervised_checkpoint/ce_distilbert-base-uncased_R8_2026-10-25-03-30-08'

train, val, test = load_data('../data/R8/')
_, _, test_loader, num_labels = prepare_dataloaders(args, train, val, test)
model = BertClassifier(pretrained_model=args.bert_init, nb_class=num_labels).to(device)

ckpt = torch.load(f'{ckpt_dir}/checkpoint.pth')
model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded epoch {ckpt['epoch']+1} (the real best checkpoint)")

logger = setup_logger(ckpt_dir)
test_evaluate(model, test_loader, logger, device)
