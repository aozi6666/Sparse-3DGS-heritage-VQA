# 离线视频依赖

* AKS 的源码与原有数据位于 AKS/。逐题候选从该目录读取 VideoMME 数据、BLIP 分数和 32 帧结果。在本目录执行 sha256sum -c AKS-videomme.sha256 可核对四个服务器输入。
* Explore-Then-Select/ 保存 ANDgate99/Explore-Then-Select 的 upstream fdd46c94af4a7819724c1cfacacccf0626d32e1a。项目原有的实验说明位于 ../Explore-Then-Select/。上游 Git 元数据另存于本机 .explore-upstream-git。
* Mage_ViT/ 保存上游源码，运行环境位于 /root/autodl-tmp/Mage_ViT/mage。
* 演示工程工作树位于 /root/autodl-tmp/heritage-demo。
