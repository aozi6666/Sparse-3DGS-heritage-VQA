# GitHub 仓库改名

目标名：**Sparse-3DGS-heritage-VQA**（已完成时可忽略本节操作步骤）。

本机未安装 `gh` 时，请在网页操作：

1. 打开 https://github.com/aozi6666/3DGS → Settings → General → Repository name  
2. 改为 `Sparse-3DGS-heritage-VQA`  
3. 本地更新 remote：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
git remote set-url origin git@github.com:aozi6666/Sparse-3DGS-heritage-VQA.git
git remote -v
```

本地目录与仓库名对齐为 `Sparse-3DGS-heritage-VQA/`；根下并列 `3DGS/`（重建）与 `VQA/`。
