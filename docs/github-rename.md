# GitHub 仓库改名

目标名：**Sparse-3DGS-heritage-VQA**

本机未安装 `gh` 时，请在网页操作：

1. 打开 https://github.com/aozi6666/3DGS → Settings → General → Repository name  
2. 改为 `Sparse-3DGS-heritage-VQA`  
3. 本地更新 remote：

```bash
cd /root/autodl-tmp/3DGS
git remote set-url origin git@github.com:aozi6666/Sparse-3DGS-heritage-VQA.git
git remote -v
```

本地文件夹可继续叫 `3DGS/`，不影响。
