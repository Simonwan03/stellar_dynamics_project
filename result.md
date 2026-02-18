## 1D-CNN
self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),      
        )

T=100
epoch 95 | AUC 0.9524 | F1 0.8361 | Acc 0.8779
epoch 96 | AUC 0.9518 | F1 0.8319 | Acc 0.8787
epoch 97 | AUC 0.9522 | F1 0.8332 | Acc 0.8804
epoch 98 | AUC 0.9486 | F1 0.8183 | Acc 0.8560
epoch 99 | AUC 0.9447 | F1 0.8173 | Acc 0.8672
epoch 100 | AUC 0.9501 | F1 0.8287 | Acc 0.8725

T=200
epoch 45 | AUC 0.9625 | F1 0.8548 | Acc 0.8866
epoch 46 | AUC 0.9594 | F1 0.8514 | Acc 0.8827
epoch 47 | AUC 0.9621 | F1 0.8514 | Acc 0.8838
epoch 48 | AUC 0.9601 | F1 0.8557 | Acc 0.8891
epoch 49 | AUC 0.9602 | F1 0.8539 | Acc 0.8874
epoch 50 | AUC 0.9591 | F1 0.8483 | Acc 0.8825
