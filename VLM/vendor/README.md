# 离线视频依赖

* AKS 的源码与原有数据位于 AKS/。逐题候选从该目录读取 VideoMME 数据、BLIP 分数和 32 帧结果。在本目录执行 sha256sum -c AKS-videomme.sha256 可核对四个服务器输入。
* Explore-Then-Select/ 保存 ANDgate99/Explore-Then-Select 的 upstream fdd46c94af4a7819724c1cfacacccf0626d32e1a。项目原有的实验说明位于 ../Explore-Then-Select/。上游 Git 元数据另存于本机 .explore-upstream-git。
* Mage_ViT/ 保存原有 Mage 源码及本机的 mage/ 虚拟环境。模型环境、上游媒体和数据留在服务器。冻结的历史协议对启动脚本逐字节校验，并把绝对路径作为来源键；服务器保留 /root/autodl-tmp/Mage_ViT 指向本目录下 Mage_ViT 的兼容链接。
* 展示工程属于 feat/project-video-deme 分支，服务器使用 ../worktrees/heritage-demo/Project。该 Git 工作树由主仓管理，保留自己的索引、前端依赖、storage 和模型链接。
